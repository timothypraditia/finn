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
path_to_data = string(abs_path, "\\data_core1.xlsx")
save_folder = "core_2"

# Load the trained parameters
@load @sprintf("%s/model.bson", save_folder) pstar


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
title("Breakthrough Curve of Core #1")
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


##############
# Prediction #
##############

# Define the ODE problem using the pre-defined network
prob_nn = ODEProblem(nn_ode, c0, (0.0, T), pstar)
pred = Array(concrete_solve(prob_nn,Tsit5(),c0,pstar,saveat=t))
breakthrough_pred = (pred[Nx-1,:] .- pred[Nx,:]) .* D .* por .* cross_area ./ dx ./ Q

# Plot the prediction
figure()
scatter(t,1000*breakthrough_data,label="Experimental Data",color="red")
breakthrough_plot = plot(t, 1000*breakthrough_pred,label="Prediction")
title("Breakthrough Curve")
xlabel(L"time [$days$]")
ylabel(L"Tailwater concentration [$mg/L$]")
legend()
gcf()

savefig(@sprintf("%s/core_1.png", save_folder))
