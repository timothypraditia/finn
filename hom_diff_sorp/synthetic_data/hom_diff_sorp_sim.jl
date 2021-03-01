cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using PyPlot
using Printf
using Flux, DiffEqFlux, Optim, DiffEqSensitivity
using DifferentialEquations
using DelimitedFiles
using BSON: @save, @load

#########################
# Set working directory #
#########################

abs_path = pwd()
save_folder = string(abs_path,"\\data_freundlich")
path_to_cdiss_data = @sprintf("%s/c_diss.txt", save_folder)
path_to_ctot_data = @sprintf("%s/c_tot.txt", save_folder)


##################
# Initialization #
##################

# Parameters
D = 0.0005; # Effective diffusion coefficient [m^2/day]
por = 0.29; # Porosity [-]
rho_s = 2880; # Dry bulk density [kg/m^3]
Kf = 1.016/rho_s; # Freundlich K
nf = 0.874; # Freundlich exponent
smax = 1/1700; # Sorption capacity [m^3/kg]
Kl = 1; # Half-concentration [kg/m^3]
Kd = 0.429/1000; # Partitioning coefficient [m^3/kg]

# Freundlich retardation factor (change accordingly)
retardation(u) = 1 .+ (1 .- por) ./ por .* rho_s .* Kf .* nf .* (u .+ 1e-6) .^ (nf-1) # freundlich

# For plotting
lin_sorp(u) = 1 .+ (1 .- por) ./ por .* rho_s .* Kd # linear
freundlich(u) = 1 .+ (1 .- por) ./ por .* rho_s .* Kf .* nf .* (u .+ 1e-6) .^ (nf-1) # freundlich
langmuir(u) = 1 .+ (1 .- por) ./ por .* rho_s .* smax .* Kl ./ (u .+ Kl).^2 # langmuir

# Simulation domain
X = 1.0; T = 2500;
dx = 0.04; dt = T/500;
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

# Set the first 501 time steps as training data
ode_data = vcat(c_diss[:,1:501],c_tot[:,1:501])

# Initial conditions
c0 = c_diss[:,1]
c_tot0 = c_tot[:,1]




## Train dissolved concentration

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

p_1 = [p1;D0;p_stencil;p_exp]

# To restructure the flattened params
full_restructure(p_1) = re1(p_1[1:length(p1)]), p_1[end-3], p_1[end-2:end-1], p_1[end]

# Define the flux kernel
function flux_kernel(u,p_1)
    # Calculate flux at the left cell connector
    left = [p_1[end-1] * u[i-1] + p_1[end-2] * u[i] for i in 2:Nx]
    # At the left boundary
    left_bc_flux = [p_1[end-1] * left_BC + p_1[end-2] * u[1]]
    left_flux = vcat(left_bc_flux, left)

    # Calculate flux at the right cell connector
    right = [p_1[end-2] * u[i] + p_1[end-1] * u[i+1] for i in 1:Nx-1]
    # At the right boundary
    right_BC = (u[Nx-1] - u[Nx]) * p_1[end-3]*dx
    right_bc_flux = [p_1[end-2] * u[Nx] + p_1[end-1] * right_BC]
    right_flux = vcat(right, right_bc_flux)

    # Integrate fluxes at all connectors
    left_flux + right_flux
end

# Define the state kernel
function state_kernel(u,p_1,flux)
    # Recall the network to calculate retardation factor
    rx_nn = re1(p_1[1:length(p1)])

    # Calculate du/dt using the flux calculated by the flux kernels
    [rx_nn([u[i]])[1] for i in 1:Nx] ./ 10 .^ p_1[end] .* flux
end

# Define the network as an ODE problem
function nn_ode(u,p_1,t)
    # Calculate fluxes with flux kernels
    flux = flux_kernel(u,p_1)

    # Calculate and return du/dt with state kernels
    state_kernel(u,p_1,flux)
end


############
# Training #
############

# Define the ODE problem using the pre-defined network
prob_nn = ODEProblem(nn_ode, c0, (0.0, T), p_1)

# Define the loss function
function loss_rd(p_1)
    # Calculate the network prediction
    pred = Array(concrete_solve(prob_nn,Tsit5(),c0,p_1,saveat=dt,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))

    # Recall the retardation factor to be regularized as a monotonically decreasing function
    rx_nn = re1(p_1[1:length(p1)])
    u = collect(0:0.01:1)
    ret = p_1[end-3] ./ [rx_nn([elem]) for elem in u] .* 10^p_1[end]

    # Define the loss function as a sum of the squared error and the retardation
    # factor regularization
    sum(abs2, ode_data[1:Nx,:] .- pred) + 10^2 * sum(relu.(ret[2:end] .- ret[1:end-1])) + 10^2 * abs(sum(p_1[end-2 : end-1])), pred, ret
end

# Define the callback function to log training
global iter = 0
global save_iter = 0
save_freq = 50

train_arr = Float64[]

cb = function (p_1,l,pred,ret)
    # Restructure the network from the flattened parameters
    rx_nn, D_pred, p_stencil, p_exp = full_restructure(p_1)

    # Record the loss value in an array and print
    push!(train_arr, l)
    println(@sprintf("Loss: %0.4f\tD: %0.4f\tWeights:(%0.4f,\t %0.4f) \t Sum: %0.4f"
            ,l, p_1[end-3], p_1[end-2], p_1[end-1], sum(p_1[end-2:end-1])))

    global iter

    # Initialize the plotting
    if iter==0
        fig = figure(figsize=(8,5.0));
        ttl = fig.suptitle(@sprintf("Epoch = %d", iter), y=1.05)
        global ttl

        # Plot the dissolved concentration data
        subplot(131)
        pcolormesh(x,t,ode_data[1:Nx,:]')
        xlabel(L"$x$"); ylabel(L"$t$"); title("Dissolved Concentration Data")
        colorbar(); clim([0, maximum(ode_data[1:Nx,:])]);

        # Plot the dissolved concentration prediction
        subplot(132)
        img1 = pcolormesh(x,t,pred[1:Nx,:]')
        global img1
        xlabel(L"$x$"); ylabel(L"$t$"); title("Dissolved Concentration Prediction")
        colorbar(); clim([0, maximum(ode_data[1:Nx,:])]);

        # Plot the retardation factor
        ax = subplot(133); global ax
        u = collect(0.01:0.01:1)
        rx_pred = D_pred ./ rx_nn.([[elem] for elem in u]) .* 10^p_1[end]
        rx_line = plot(u, rx_pred, label="NN")[1];
        global rx_line
        plot(u, retardation.(u), label="True")
        title("Retardation Factor")
        legend(loc="upper right", frameon=false, fontsize=8);
        min_lim = min(minimum(rx_pred), minimum(retardation.(u)))*0.9
        max_lim = max(maximum(rx_pred), maximum(retardation.(u)))*1.1
        ylim([min_lim, max_lim])

        subplots_adjust(top=0.8)
        tight_layout()
    end

    # Updating the plots
    if iter>0
        # Update the prediction plot
        println("updating figure")

        img1.set_array(pred[1:end-1,1:end-1][:])
        ttl.set_text(@sprintf("Epoch = %d", iter))

        u = collect(0.01:0.01:1)
        rx_pred = D_pred ./ rx_nn.([[elem] for elem in u]) .* 10^p_1[end]
        rx_line.set_ydata(rx_pred)

        min_lim = min(minimum(rx_pred), minimum(retardation.(u)))*0.9
        max_lim = max(maximum(rx_pred), maximum(retardation.(u)))*1.1

        ax.set_ylim([min_lim, max_lim])
    end

    # Save the training plots
    global save_iter

    if iter%save_freq == 0
        savefig(@sprintf("%s/diss_pred_%05d.png", save_folder, save_iter), dpi=200, bbox_inches="tight")
        save_iter += 1
    end

    display(gcf())
    iter += 1

    false
end

# Train first with ADAM and continue with BFGS
res_diss1 = DiffEqFlux.sciml_train(loss_rd, p_1, ADAM(0.001), cb=cb, maxiters = 400)
res_diss2 = DiffEqFlux.sciml_train(loss_rd, res_diss1.minimizer, BFGS(), cb=cb, maxiters = 100)

# Update and save plot
global save_iter
println("saved figure")
savefig(@sprintf("%s/diss_pred_%05d.png", save_folder, save_iter), dpi=200, bbox_inches="tight")
save_iter += 1

# Save the parameters after training with dissolved concentration
pstar_1 = res_diss2.minimizer
@save @sprintf("%s/model_diss.bson", save_folder) pstar_1


## Train total concentration

# Define a new trainable parameter which is the diffusion coefficient
p_2 = [pstar_1[end-3]]

# Redefine flux kernel
function flux_kernel(u,p_2)
    # Calculate flux at the left cell connector
    left = [pstar_1[end-1] * u[i-1] + pstar_1[end-2] * u[i] for i in 2:Nx]
    # At the left boundary
    left_bc_flux = [pstar_1[end-1] * left_BC + pstar_1[end-2] * u[1]]
    left_flux = vcat(left_bc_flux, left)

    # Calculate flux at the right cell connector
    right = [pstar_1[end-2] * u[i] + pstar_1[end-1] * u[i+1] for i in 1:Nx-1]
    # At the right boundary
    right_BC = (u[Nx-1] - u[Nx]) * p_2[1]*dx
    right_bc_flux = [pstar_1[end-2] * u[Nx] + pstar_1[end-1] * right_BC]
    right_flux = vcat(right, right_bc_flux)

    # Integrate fluxes at all connectors
    left_flux + right_flux
end

# Redefine state kernel
function state_kernel(u,p_2,flux)
    # Recall the network to calculate retardation factor
    rx_nn = re1(pstar_1[1:length(p1)])

    # Calculate du/dt using the flux calculated by the flux kernels
    dc = [rx_nn([u[i]])[1] for i in 1:Nx] ./ 10 .^ pstar_1[end] .* flux
    dc_tot = p_2[1] * por / (rho_s/1000) * flux
    vcat(dc,dc_tot)
end

# Redefine the network as an ODE problem
function nn_ode(u,p_2,t)
    # Calculate fluxes with flux kernels
    flux = flux_kernel(u,p_2)

    # Calculate and return du/dt with state kernels
    state_kernel(u,p_2,flux)
end


############
# Training #
############

# Define the ODE problem using the pre-defined network
prob_nn = ODEProblem(nn_ode, vcat(c0,c_tot0), (0.0, T), p_2)

# Define the loss function
function loss_rd(p_2)
    # Calculate the network prediction
    pred = Array(concrete_solve(prob_nn,Tsit5(),vcat(c0,c_tot0),p_2,saveat=dt,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))

    # Define the loss function as a sum of the squared error
    sum(abs2, ode_data .- pred), pred
end

# Define the callback function to log training
global reset_train = true
global save_iter = 0

cb = function (p_2,l,pred)
    # Restructure the network from the flattened parameters
    rx_nn = re1(pstar_1[1:length(p1)])

    # Record the loss value in an array and print
    push!(train_arr, l)
    println(@sprintf("Loss: %0.4f\tD: %0.4f\tWeights:(%0.4f,\t %0.4f) \t Sum: %0.4f"
            ,l, p_2[1], pstar_1[end-2], pstar_1[end-1], sum(pstar_1[end-2:end-1])))

    global iter
    global reset_train

    # Initialize the plotting
    if reset_train
        fig = figure(figsize=(8,5.0));
        ttl = fig.suptitle(@sprintf("Epoch = %d", iter), y=1.05)
        global ttl

        # Plot the total concentration data
        subplot(131)
        pcolormesh(x,t,ode_data[Nx+1:2*Nx,:]')
        xlabel(L"$x$"); ylabel(L"$t$"); title("Total Concentration Data")
        colorbar(); clim([0, maximum(ode_data[Nx+1:2*Nx,:])]);

        # Plot the total concentration prediction
        subplot(132)
        img1 = pcolormesh(x,t,pred[Nx+1:2*Nx,:]')
        global img1
        xlabel(L"$x$"); ylabel(L"$t$"); title("Total Concentration Prediction")
        colorbar(); clim([0, maximum(ode_data[Nx+1:2*Nx,:])]);

        # Plot the retardation factor
        ax = subplot(133); global ax
        u = collect(0.01:0.01:1)
        rx_pred = p_2[1] ./ rx_nn.([[elem] for elem in u]) .* 10^pstar_1[end]
        rx_line = plot(u, rx_pred, label="NN")[1];
        global rx_line
        plot(u, retardation.(u), label="True")
        title("Retardation Factor")
        legend(loc="upper right", frameon=false, fontsize=8);
        min_lim = min(minimum(rx_pred), minimum(retardation.(u)))*0.9
        max_lim = max(maximum(rx_pred), maximum(retardation.(u)))*1.1
        ylim([min_lim, max_lim])

        subplots_adjust(top=0.8)
        tight_layout()

        reset_train = false

    else
        # Update the prediction plot
        println("updating figure")

        img1.set_array(pred[Nx+1:end-1,1:end-1][:])
        ttl.set_text(@sprintf("Epoch = %d", iter))

        u = collect(0.01:0.01:1)
        rx_pred = p_2[1] ./ rx_nn.([[elem] for elem in u]) .* 10^pstar_1[end]
        rx_line.set_ydata(rx_pred)

        min_lim = min(minimum(rx_pred), minimum(retardation.(u)))*0.9
        max_lim = max(maximum(rx_pred), maximum(retardation.(u)))*1.1

        ax.set_ylim([min_lim, max_lim])
    end

    # Save the training plots
    global save_iter

    if iter%save_freq == 0
        println("saved figure")
        savefig(@sprintf("%s/tot_pred_%05d.png", save_folder, save_iter), dpi=200, bbox_inches="tight")
        save_iter += 1
    end

    display(gcf())
    iter += 1

    false
end

# Train with BFGS
res_tot = DiffEqFlux.sciml_train(loss_rd, p_2, BFGS(), cb=cb, maxiters = 100)
pstar_2 = [pstar_1[1:length(p1)]; res_tot.minimizer; pstar_1[end-2:end]]

# Update and save plot
global save_iter
println("saved figure")
savefig(@sprintf("%s/tot_pred_%05d.png", save_folder, save_iter), dpi=200, bbox_inches="tight")
save_iter += 1

# Save the parameters after training with total concentration
pstar_1 = res_diss2.minimizer
@save @sprintf("%s/model_tot.bson", save_folder) pstar_2


## Plot Results

rx_nn, D_pred, p_stencil, p_exp = full_restructure(pstar_2)

# Redefine the Neural Network
function flux_kernel(u,p_star2)
    left = [pstar_2[end-1] * u[i-1] + pstar_2[end-2] * u[i] for i in 2:Nx]
    left_bc_flux = [pstar_2[end-1] * left_BC + pstar_2[end-2] * u[1]]
    left_flux = vcat(left_bc_flux, left)

    right = [pstar_2[end-2] * u[i] + pstar_2[end-1] * u[i+1] for i in 1:Nx-1]
    right_BC = (u[Nx-1] - u[Nx]) * pstar_2[end-3]*dx
    right_bc_flux = [pstar_2[end-2] * u[Nx] + pstar_2[end-1] * right_BC]
    right_flux = vcat(right, right_bc_flux)

    left_flux + right_flux
end

function state_kernel(u,pstar_2,flux)
    rx_nn = re1(pstar_2[1:length(p1)])

    dc = [rx_nn([u[i]])[1] for i in 1:Nx] ./ 10 .^ pstar_2[end] .* flux
    dc_tot = pstar_2[end-3] * por / (rho_s/1000) * flux
    vcat(dc,dc_tot)
end

function nn_ode(u,pstar_2,t)
    flux = flux_kernel(u,pstar_2)

    state_kernel(u,pstar_2,flux)
end

# Calculate the prediction
prob_nn = ODEProblem(nn_ode, vcat(c0,c_tot0), (0.0, T), pstar_2)
pred = Array(concrete_solve(prob_nn,Tsit5(),vcat(c0,c_tot0),pstar_2,saveat=dt))

# Restructure the network to calculate retardation factor
rx_nn, D_pred, p_stencil, p_exp = full_restructure(pstar_2)

# Plot the results
fig = figure(figsize=(6,6))

# Dissolved concentration data
subplot(321)
pcolormesh(x,t,ode_data[1:Nx,:]', rasterized=true)
xlabel(L"$x$"); ylabel(L"$t$"); title("Dissolved Concentration Data")
yticks([0, 500, 1000, 1500, 2000, 2500])
colorbar(); clim([0, maximum(ode_data[1:Nx,:])]);

# Dissolved concentration prediction
ax = subplot(322)
pcolormesh(x,t,pred[1:Nx,:]', rasterized=true)
xlabel(L"$x$"); ylabel(L"$t$"); title("Dissolved Concentration Prediction")
yticks([0, 500, 1000, 1500, 2000, 2500])
colorbar(); clim([0, maximum(ode_data[1:Nx,:])]);

# Total concentration data
subplot(323)
pcolormesh(x,t,ode_data[Nx+1:2*Nx,:]', rasterized=true)
xlabel(L"$x$"); ylabel(L"$t$"); title("Total Concentration Data")
yticks([0, 500, 1000, 1500, 2000, 2500])
colorbar(); clim([0, maximum(ode_data[Nx+1:2*Nx,:])]);

# Total concentration prediction
ax = subplot(324)
pcolormesh(x,t,pred[Nx+1:2*Nx,:]', rasterized=true)
xlabel(L"$x$"); ylabel(L"$t$"); title("Total Concentration Prediction")
yticks([0, 500, 1000, 1500, 2000, 2500])
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
savefig(@sprintf("%s/train_result.png", save_folder))

# Plot loss vs epochs and save
figure(figsize=(6,3))
plot(log.(train_arr), "k.", markersize=1)
xlabel("Epochs"); ylabel("Log(loss)")
tight_layout()
savefig(@sprintf("%s/loss_vs_epoch.pdf", save_folder))
gcf()
