cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using XLSX
using PyPlot
using Printf
using Flux, DiffEqFlux, Optim, DiffEqSensitivity
using DifferentialEquations
using BSON: @save, @load

#########################
# Set working directory #
#########################

abs_path = pwd()
path_to_data = string(abs_path, "\\data_core2.xlsx")
save_folder = "core_2"

if ~isdir(save_folder)
    mkdir(save_folder)
end


#######################
# Read data from file #
#######################

# Breakthrough data and time steps
xf1 = XLSX.readxlsx(path_to_data)[1]
all_data = xf1[:]

t = convert(Array{Float64,1},all_data[:,1]) #days
breakthrough_data = convert(Array{Float64,1},all_data[:,2])/1000 #kg/m^3

# Soil parameters
xf2 = XLSX.readxlsx(path_to_data)[2]
params = xf2[:]

D = params[findfirst(isequal("D"),params)[1],2]; #effective diffusion [m^2/day]
por = params[findfirst(isequal("por"),params)[1],2]; # porosity [-]
rho_s = params[findfirst(isequal("rho_s"),params)[1],2]; #dry bulk density, [kg/m^3]

# Simulation domain
X = params[findfirst(isequal("X"),params)[1],2] #[m];
T = t[end]; #[days]
Nx = params[findfirst(isequal("Nx"),params)[1],2];
Nt = size(t)[1];
dx = X/(Nx+1);
x = collect(dx:dx:X-dx);
sample_radius = params[findfirst(isequal("sample_radius"),params)[1],2]; #[m]
cross_area = π*sample_radius^2; #cross sectional area [m^2]
Q = params[findfirst(isequal("Q"),params)[1],2]; #flow rate in bottom reservoir [m^3/day]

# Boundary condition
solubility = params[findfirst(isequal("solubility"),params)[1],2]; #constant upper boundary condition [kg/m^3]
Dirichlet = Bool(params[findfirst(isequal("Dirichlet"),params)[1],2])
Cauchy = Bool(params[findfirst(isequal("Cauchy"),params)[1],2])

#Initial conditions
c0 = zeros(Nx)
global right_BC = 0;

# Plot the data
figure()
scatter(t,breakthrough_data)
title("Breakthrough Curve of Core #2")
xlabel(L"time [$days$]")
ylabel(L"Tailwater concentration [$mg/L$]")
gcf()


######################
# Define the network #
######################

# Define the network to learn the retardation factor
n_weights = 10
rx_nn = Chain(Dense(1, n_weights, tanh),
                Dense(n_weights, 2*n_weights, tanh),
                Dense(2*n_weights, n_weights, tanh),
                Dense(n_weights, 1, σ),
                x -> x[1])

# For parameters flattening
p1,re1 = Flux.destructure(rx_nn)
p_exp = [1.0]
p = [p1; p_exp]

# To restructure the flattened params
full_restructure(p) = re1(p[1:length(p1)]), p[end]

# Define the flux kernel
function flux_kernel(u,p)
    # Set the variable right_BC to be global to extract value of the Cauchy BC
    # at the previous time step
    global right_BC

    # Set right_BC = 0 if Dirichlet
    if Dirichlet
        right_BC = 0
    end

    # Reset right_BC = 0 for initial condition, otherwise calculate the Cauchy BC
    if Cauchy
        if u == c0
            right_BC = 0
        end
        right_BC = (u[Nx]-right_BC)/dx * D * por * cross_area / Q
    end

    # Calculate flux at the left cell connector
    left = [1 * u[i-1] - 1 * u[i] for i in 2:Nx]
    # At the left boundary
    left_bc_flux = [1 * solubility - 1 * u[1]]
    left_flux = vcat(left_bc_flux, left)

    # Calculate flux at the right cell connector
    right = [- 1 * u[i] + 1* u[i+1] for i in 1:Nx-1]
    # At the right boundary
    right_bc_flux = [- 1 * u[Nx] + 1 * right_BC]
    right_flux = vcat(right, right_bc_flux)

    # Integrate fluxes at all connectors
    left_flux + right_flux
end

# Define the state kernel
function state_kernel(u,p,flux)
    # Recall the network to calculate retardation factor
    rx_nn = re1(p[1:length(p1)])

    # Calculate du/dt using the flux calculated by the flux kernels
    D / dx^2 ./ (1.0 .+ [rx_nn([u[i]])[1] for i in 1:Nx] .* 10 .^ p[end]) .* flux
end

# Define the network as an ODE problem
function nn_ode(u,p,t)
    # Calculate fluxes with flux kernels
    flux = flux_kernel(u,p)

    # Calculate and return du/dt with state kernels
    state_kernel(u,p,flux)
end


#####################
# Train the network #
#####################

# Define the ODE problem using the pre-defined network
prob_nn = ODEProblem(nn_ode, c0, (0.0, T), p)

# Define the loss function
function loss_rd(p)
    # Calculate the network prediction
    pred = Array(concrete_solve(prob_nn,Tsit5(),c0,p,saveat=t,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))

    # From the network prediction, calculate the tailwater concentration
    breakthrough_pred = (pred[Nx-1,:] .- pred[Nx,:]) .* D .* por .* cross_area ./ dx ./ Q

    # Recall the retardation factor to be regularized as a monotonically decreasing function
    rx_nn = re1(p[1:length(p1)])
    u = collect(0:0.001:2.0)
    ret = 1.0 .+ [rx_nn([elem]) for elem in u] .* 10^p[end]

    # Define the loss function as a sum of the squared error and the retardation
    # factor regularization
    sum(abs2, 10^3*breakthrough_data .- 10^3*breakthrough_pred) + 10^2 * sum(relu.(ret[2:end] .- ret[1:end-1])), pred, breakthrough_pred, ret
end

# Define the callback function to log training
global iter = 0
global save_iter = 0
save_freq = 50

train_arr = Float64[]

cb = function (p,l,pred,breakthrough_pred,ret)
    # Restructure the network from the flattened parameters
    rx_nn, p_exp = full_restructure(p)

    # Record the loss value in an array and print
    push!(train_arr, l)
    println(@sprintf("Loss: %0.4f",l))

    global iter

    # Initialize the plotting
    if iter==0
        # Plot the breakthrough curve with the data and prediction
        fig = figure(figsize=(8,5.0));
        ttl = fig.suptitle(@sprintf("Epoch = %d", iter), y=1.05)
        global ttl
        subplot(121)
        scatter(t, 1000*breakthrough_data,label="Experimental Data",color="red")
        breakthrough_plot = plot(t, 1000*breakthrough_pred,label="Prediction")[1];
        global breakthrough_plot
        title("Breakthrough Curve")
        xlabel(L"time [$days$]")
        ylabel(L"Tailwater concentration [$mg/L$]")
        legend()

        # Plot the retardation factor as a function of the concentration
        ax2 = subplot(122); global ax2
        u = collect(0:0.001:2.0)
        rx_line = plot(u, ret, label="NN")[1];
        global rx_line
        title("Retardation Factor")
        xlabel(L"$c_{diss}$")
        ylabel(L"$R$")
        min_lim = minimum(ret)*0.9
        max_lim = maximum(ret)*1.1
        ylim([min_lim, max_lim])

        subplots_adjust(top=0.8)
        tight_layout()
    end

    # Updating the plots
    if iter>0
        println("updating figure")

        # Update the prediction plot
        ttl.set_text(@sprintf("Epoch = %d", iter))
        breakthrough_plot.set_ydata(1000*breakthrough_pred)

        rx_line.set_ydata(ret)

        min_lim = minimum(ret)*0.9
        max_lim = maximum(ret)*1.1

        ax2.set_ylim([min_lim, max_lim])
    end

    # Save the training plots
    global save_iter

    if iter%save_freq == 0
        println("saved figure")
        savefig(@sprintf("%s/diss_pred_%05d.png", save_folder, save_iter), dpi=200, bbox_inches="tight")
        save_iter += 1
    end

    display(gcf())
    iter += 1

    false
end

# Train first with ADAM and continue with BFGS
res1 = DiffEqFlux.sciml_train(loss_rd, p, ADAM(0.001), cb=cb, maxiters = 100)
res2 = DiffEqFlux.sciml_train(loss_rd, res1.minimizer, BFGS(), cb=cb, maxiters = 1000)

# Save the optimized parameters
pstar = res2.minimizer
@save @sprintf("%s/model.bson", save_folder) pstar
