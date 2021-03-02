cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

#This script simulates the non-linear diffusion-sorption equation and fits
#a finite volume neural PDE to the data

using PyPlot, Printf
using LinearAlgebra
using BSON: @save, @load
using DifferentialEquations
using DelimitedFiles

# Parameters
D = 0.0005; # Effective diffusion coefficient [m^2/day]
por = 0.29; # Porosity [-]
rho = 2880; # Dry bulk density [kg/m^3]
Kf = 1.016/rho; # Freundlich K
nf = 0.874; # Freundlich exponent

# Freundlich retardation factor
retardation(u) = 1 .+ (1 .- por) ./ por .* rho .* Kf .* nf .* (u .+ 1e-6) .^ (nf-1) # freundlich

# For plotting
lin_sorp(u) = 1 .+ (1 .- por) ./ por .* rho .* Kd # linear
freundlich(u) = 1 .+ (1 .- por) ./ por .* rho .* Kf .* nf .* (u .+ 1e-6) .^ (nf-1) # freundlich
langmuir(u) = 1 .+ (1 .- por) ./ por .* rho .* smax .* Kl ./ (u .+ Kl).^2 # langmuir

# Simulation domain
X = 1.0; T = 10000;
dx = 0.04; dt = T/1000;
x = collect(0:dx:X);
t = collect(0:dt:T);
Nx = Int64(X/dx+1);
Nt = Int64(T/dt+1);

# Initial conditions
c0 = zeros(Nx)
c_tot0 = zeros(Nx)

# Boundary conditions
solubility = 1.0

save_folder = "data_freundlich"

########################
# Generate training data
########################
lap = diagm(0 => -2.0 * ones(Nx), 1=> ones(Nx-1), -1 => ones(Nx-1)) ./ dx^2
q = zeros(Nx)
q_tot = zeros(Nx)

function rc_ode(c, p, t)
    left_BC = solubility
    right_BC = (c[Nx-1]-c[Nx])/dx * D

    q[1] = D / retardation(c[1]) / dx^2 * left_BC
    q[end] = D / retardation(c[Nx]) / dx^2 * right_BC
    dc = D ./ retardation.(c[1:Nx]) .* lap * c[1:Nx] + q

    q_tot[1] = D * por / (rho/1000) / dx^2 * left_BC
    q_tot[end] = D * por / (rho/1000) / dx^2 * right_BC
    dc_tot = D * por / (rho/1000) .* lap * c[1:Nx] + q_tot

    vcat(dc,dc_tot)
end

prob = ODEProblem(rc_ode, vcat(c0,c_tot0), (0.0, T), saveat=dt)
sol = solve(prob, Tsit5());
data = Array(sol);

fig = figure(figsize=(8,6))

subplot(121)
pcolormesh(x,t,data[1:Nx,:]', rasterized=true)
xlabel(L"$x$"); ylabel(L"$t$"); title("Dissolved Concentration")
yticks([0, 2000, 4000, 6000, 8000, 10000])
colorbar(); clim([0, 1]);

ax = subplot(122)
pcolormesh(x,t,data[Nx+1:end,:]', rasterized=true)
xlabel(L"$x$"); ylabel(L"$t$"); title("Total Concentration")
yticks([0, 2000, 4000, 6000, 8000, 10000])
colorbar(); clim([0, maximum(data[Nx+1:end,:])]);

tight_layout()
gcf()

writedlm(@sprintf("%s/c_diss.txt", save_folder), data[1:Nx,:])
writedlm(@sprintf("%s/c_tot.txt", save_folder), data[Nx+1:end,:])
