using LinearAlgebra
using StatsBase
using CairoMakie
using JSON

##################################################################################################


function w_matrix(memories::Matrix{Float64})
    
    # returns a W matrix trained to retrive memories
    # get sizes for memoeries and list of memories
    
    P,N = size(memories)
 
    W = zeros(N,N)
    
    for i in 1:N
        for j in 1:N
            if i != j
                W[i,j] = (1/P)*sum([memories[mu,i]*memories[mu,j] for mu in 1:P])
            end
        end
    end
    
    return W
end


function update_state(state::Vector{Float64}, overlaps::Vector{Float64}, patterns::Matrix{Float64})
    
    N = length(state)
    P = length(overlaps)
    new_state = zeros(N)
    
    for i in 1:N
        h_i = 0
        for mu in 1:P
            h_i += patterns[mu,i]*N*overlaps[mu] - P*state[i]
        end
        new_state[i] = sign(h_i)
    end
    
    return new_state
end

function overlap(mem_1::Vector{Float64}, mem_2::Vector{Float64})
    
    N_1 = length(mem_1)
    N_2 = length(mem_2)
    @assert N_1 == N_2 "Memories sizes dont match"
    
    overlap = (1/N_1)*sum([mem_1[i]*mem_2[i] for i in 1:N_1])
    return overlap
end

function gen_random_memory(size::Int64)
    mem = zeros(size)
    for i in 1:size
        p = rand()
        if p < 0.5
            mem[i] = -1
        else
            mem[i] = 1
        end
    end
    return mem
end

function noise_memory(p::Float64, memory::Vector{Float64})
    
    new_memory = Float64[]
    
    for (i,s_i) in enumerate(memory)
        r = rand()
        if r < p
            push!(new_memory, -1*s_i)
        else
            push!(new_memory, s_i)
        end
    end
    
    return new_memory
    
end

########################################################################################

# Leemos los datos del cmd

N = parse(Int64, ARGS[1])
P = parse(Int64, ARGS[2])
N_iter =  parse(Int64, ARGS[3])
p_noise = parse(Float64, ARGS[4])
folder = ARGS[5]

##############################################################

initial_memories = zeros(P,N)
state_memories = zeros(P,N)


for mu in 1:P
    initial_memories[mu,:] = gen_random_memory(N)
end

for mu in 1:P
    state_memories[mu,:] = noise_memory(p_noise, initial_memories[mu,:])
end


W = w_matrix(initial_memories)
overlap_dict = Dict()


for mu in 1:P
    overlap_dict[mu] = Float64[]
end

##############################################################


for i in 1:N_iter
    
    for mu in 1:P

        mem_mu = state_memories[mu,:]

        overlaps = Float64[]

        for nu in 1:P
            mem_nu = initial_memories[nu,:]
            ovlp_mu_nu = overlap(mem_mu, mem_nu)
            push!(overlaps, ovlp_mu_nu)
        end

        state_memories[mu,:] = sign.(W*mem_mu)
        
        
        # solo guardamos los ultimos 20 overlaps
        
        if i > (N_iter - 20)
            ovl_max = maximum(overlaps)
            push!(overlap_dict[mu], ovl_max)
        end
        
    end
end

#######################################################################

# guardamos los resulatdos

f = open("$(folder)/overlap_$(N)_$(P).json","w")
JSON.print(f,overlap_dict) 
close(f) 