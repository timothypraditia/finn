cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using PyPlot, Printf
using DifferentialEquations
using Flux, DiffEqFlux, Optim, DiffEqSensitivity
using DelimitedFiles
using BSON: @save, @load

#########################
# Set working directory #
#########################

abs_path = pwd()
save_folder = string(abs_path,"\\data_freundlich")
path_to_cdiss_data = @sprintf("%s/c_diss.txt", save_folder)
path_to_ctot_data = @sprintf("%s/c_tot.txt", save_folder)

# Load the trained parameters
@load @sprintf("%s/model_tot.bson", save_folder) pstar_2


##################
# Initialization #
##################

# Parameters
D = 0.0005; # Effective diffusion coefficient [m^2/day]
por = 0.29; # Porosity [-]
rho_s = 2880; # Dry bulk density [kg/m^3]
Kf = 1.016/rho; # Freundlich K
nf = 0.874; # Freundlich exponent
smax = 1/1700; # Sorption capacity [m^3/kg]
Kl = 1; # Half-concentration [kg/m^3]
Kd = 0.429/1000; # Partitioning coefficient [m^3/kg]

# Freundlich retardation factor (change accordingly)
retardation(u) = 1 .+ (1 .- por) ./ por .* rho .* Kf .* nf .* (u .+ 1e-6) .^ (nf-1) # freundlich

# For plotting
lin_sorp(u) = 1 .+ (1 .- por) ./ por .* rho .* Kd # linear
freundlich(u) = 1 .+ (1 .- por) ./ por .* rho .* Kf .* nf .* (u .+ 1e-6) .^ (nf-1) # freundlich
langmuir(u) = 1 .+ (1 .- por) ./ por .* rho .* smax .* Kl ./ (u .+ Kl).^2 # langmuir

# Simulation domain
X = 1.0; T = 10000;
dx = 0.04; dt = T/2000;
x = collect(0:dx:X);
t = collect(0:dt:T);
Nx = Int64(X/dx+1);
Nt = Int64(T/dt+1);

# Boundary conditions
left_BC = 1.0 # [kg/m^3]


#############
# Read Data #
#############

c_diss = readdlm(path_to_cdiss_data)
c_tot = readdlm(path_to_ctot_data)
ode_data = vcat(c_diss,c_tot)

# Initial conditions
c0 = c_diss[:,1]
c_tot0 = c_tot[:,1]


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
p_stencil = [-1.1, 1.05]
p_exp = [1.0]
D0 = [0.5]
p1,re1 = Flux.destructure(rx_nn)

p = [p1;D0;p_stencil;p_exp]

# To restructure the flattened params
full_restructure(p) = re1(p[1:length(p1)]), p[end-3], p[end-2:end-1], p[end]

# Define the flux kernel
function flux_kernel(u,p)
    # Calculate flux at the left cell connector
    left = [p[end-1] * u[i-1] + p[end-2] * u[i] for i in 2:Nx]
    # At the left boundary
    left_bc_flux = [p[end-1] * left_BC + p[end-2] * u[1]]
    left_flux = vcat(left_bc_flux, left)

    # Calculate flux at the right cell connector
    right = [p[end-2] * u[i] + p[end-1] * u[i+1] for i in 1:Nx-1]
    # At the right boundary
    right_BC = (u[Nx-1] - u[Nx]) * p[end-3]*dx
    right_bc_flux = [p[end-2] * u[Nx] + p[end-1] * right_BC]
    right_flux = vcat(right, right_bc_flux)

    # Integrate fluxes at all connectors
    left_flux + right_flux
end

# Define the state kernel
function state_kernel(u,p,flux)
    # Recall the network to calculate retardation factor
    rx_nn = re1(p[1:length(p1)])

    # Calculate du/dt using the flux calculated by the flux kernels
    dc = [rx_nn([u[i]])[1] for i in 1:Nx] ./ 10 .^ p[end] .* flux
    dc_tot = p[end-3] * por / (rho_s/1000) * flux
    vcat(dc, dc_tot)
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
prob_nn = ODEProblem(nn_ode, vcat(c0,c_tot0), (0.0, T), pstar_2)
pred = Array(concrete_solve(prob_nn,Tsit5(),vcat(c0,c_tot0),pstar_2,saveat=t))

# Calculate the normalized MSE
norm_data = (ode_data .- minimum(ode_data)) ./ (maximum(ode_data) .- minimum(ode_data))
norm_pred = (pred .- minimum(ode_data)) ./ (maximum(ode_data) .- minimum(ode_data))
norm_mse = sum(abs2, norm_data .- norm_pred)/(size(norm_data)[1]*size(norm_data)[2])

# Restructure the network to calculate retardation factor
rx_nn, D_pred, p_stencil, p_exp = full_restructure(pstar_2)

# Plot the results
fig = figure(figsize=(6,6))

# Dissolved concentration data
subplot(321)
pcolormesh(x,t,ode_data[1:Nx,:]', rasterized=true)
xlabel(L"$x$"); ylabel(L"$t$"); title("Dissolved Concentration Data")
yticks([0, 2000, 4000, 6000, 8000, 10000])
colorbar(); clim([0, maximum(ode_data[1:Nx,:])]);

# Dissolved concentration prediction
ax = subplot(322)
pcolormesh(x,t,pred[1:Nx,:]', rasterized=true)
xlabel(L"$x$"); ylabel(L"$t$"); title("Dissolved Concentration Prediction")
yticks([0, 2000, 4000, 6000, 8000, 10000])
colorbar(); clim([0, maximum(ode_data[1:Nx,:])]);

# Total concentration data
subplot(323)
pcolormesh(x,t,ode_data[Nx+1:2*Nx,:]', rasterized=true)
xlabel(L"$x$"); ylabel(L"$t$"); title("Total Concentration Data")
yticks([0, 2000, 4000, 6000, 8000, 10000])
colorbar(); clim([0, maximum(ode_data[Nx+1:2*Nx,:])]);

# Total concentration prediction
ax = subplot(324)
pcolormesh(x,t,pred[Nx+1:2*Nx,:]', rasterized=true)
xlabel(L"$x$"); ylabel(L"$t$"); title("Total Concentration Prediction")
yticks([0, 2000, 4000, 6000, 8000, 10000])
colorbar(); clim([0, maximum(ode_data[Nx+1:2*Nx,:])]);

# Retardation factor
subplot(325)
u = collect(0.01:0.01:1)
rx_pred = D / dx^2 ./ rx_nn.([[elem] for elem in u]) .* 10^pstar_2[end] / pstar_2[end-1]
plot(u, rx_pred, label="NN")[1];
plot(u, lin_sorp.(u), linestyle="--", label="Linear")
plot(u, freundlich.(u), linestyle="--", label="Freundlich")
plot(u, langmuir.(u), linestyle="--", label="Langmuir")
xlabel(L"$R$")
title("Retardation Factor")
legend(loc="lower center", frameon=false, fontsize=6);
min_lim = min(minimum(rx_pred), minimum(lin_sorp.(u)), minimum(freundlich.(u)), minimum(langmuir.(u)))*0.9
max_lim = max(maximum(rx_pred), maximum(lin_sorp.(u)), maximum(freundlich.(u)), maximum(langmuir.(u)))*1.1
ylim([min_lim, max_lim])

tight_layout(h_pad=1)
gcf()
savefig(@sprintf("%s/test_result.png", save_folder))
