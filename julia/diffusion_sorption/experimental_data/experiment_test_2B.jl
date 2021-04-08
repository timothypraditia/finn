cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using XLSX
using PyPlot
using Printf
using Flux, DiffEqFlux, Optim, DiffEqSensitivity
using DifferentialEquations
using Interpolations
using BSON: @save, @load

#########################
# Set working directory #
#########################

abs_path = pwd()
path_to_data = string(abs_path, "\\data_core2_long.xlsx")
save_folder = "core_2"

# Load the trained parameters
@load @sprintf("%s/model.bson", save_folder) pstar


#######################
# Read data from file #
#######################

# Profile data and depth
xf1 = XLSX.readxlsx(path_to_data)[1]
all_data = xf1[:]

x_data = convert(Array{Float64,1},all_data[:,1]) #days
profile_data = convert(Array{Float64,1},all_data[:,2])/1000 #kg/m^3

# Profile from physical model and depth
xf3 = XLSX.readxlsx(path_to_data)[3]
all_model = xf3[:]

x_fitmodel = convert(Array{Float64,1},all_model[:,1]) #days
profile_fitmodel = convert(Array{Float64,1},all_model[:,2])/1000 #kg/m^3

# Soil parameters
xf2 = XLSX.readxlsx(path_to_data)[2]
params = xf2[:]

D = params[findfirst(isequal("D"),params)[1],2]; #effective diffusion [m^2/day]
por = params[findfirst(isequal("por"),params)[1],2]; # porosity [-]
rho_s = params[findfirst(isequal("rho_s"),params)[1],2]; #dry bulk density, [kg/m^3]

# Simulation domain
X = params[findfirst(isequal("X"),params)[1],2] #[m];
T = params[findfirst(isequal("T"),params)[1],2]; #[days]
Nx = params[findfirst(isequal("Nx"),params)[1],2];
Nt = params[findfirst(isequal("Nt"),params)[1],2];
dx = X/(Nx+1);
dt = T/(Nt-1);
x = collect(dx:dx:X-dx);
t = collect(0:dt:T);

# Boundary condition
solubility = params[findfirst(isequal("solubility"),params)[1],2]; #constant upper boundary condition [kg/m^3]
Dirichlet = Bool(params[findfirst(isequal("Dirichlet"),params)[1],2])
Cauchy = Bool(params[findfirst(isequal("Cauchy"),params)[1],2])

#Initial conditions
c0 = zeros(Nx)
global right_BC = 0;


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

# Restructure the network to calculate retardation factor
rx_nn, p_exp = full_restructure(pstar)
u = pred[:,end]
profile_model = u .* (1.0 .+ [rx_nn([elem]) for elem in u] .* 10 .^ pstar[end]) * por / (rho_s / 1000)

# Plot the prediction
figure()
scatter(x_data,1000*profile_data,color="red",label="Experimental Data")
plot(x,1000*profile_model,label="NN Prediction")
plot(x_fitmodel,1000*profile_fitmodel,linestyle="--",label="Physical Model")
title("Soil Data of Core #2B")
xlabel(L"Depth [$m$]")
ylabel(L"TCE Concentration [$mg/L$]")
legend()
gcf()

savefig(@sprintf("%s/core_2B.png", save_folder))

# Interpolate the data points to calculate MSE (the depth slices from the experimental data is slightly different)
x_data_interp = x_data[2:end-1]
interp_model = interpolate((x,),profile_model,Gridded(Linear()))
profile_model = interp_model(x_data_interp)
profile_data = profile_data[2:end-1]

# Calculate the normalized MSE of the NN prediction and calibrated physical model
norm_data = (profile_data .- minimum(profile_data))/(maximum(profile_data) .- minimum(profile_data))
norm_model = (profile_model .- minimum(profile_data))/(maximum(profile_data) .- minimum(profile_data))
norm_fitmodel = (profile_fitmodel .- minimum(profile_data))/(maximum(profile_data) .- minimum(profile_data))

mse_model = sum(abs2, norm_model .- norm_data)/size(profile_data)[1]
mse_fitmodel = sum(abs2, norm_fitmodel .- norm_data)/size(profile_data)[1]
