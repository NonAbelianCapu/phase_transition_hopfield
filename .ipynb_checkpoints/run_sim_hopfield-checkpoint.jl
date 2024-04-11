using LinearAlgebra
using StatsBase
using CairoMakie
using JSON

##################################################################################################


function w_matrix(memories::Matrix{Float64})
    # Returns a W matrix trained to retrieve memories.
    
    # Get sizes for memories matrix
    P,N = size(memories)
 
    # Initialize W matrix as a zero matrix
    W = zeros(N,N)
    
    # Iterate over dimensions of W matrix
    for i in 1:N
        for j in 1:N
            
            # Calculate W[i,j] using Hebbian learning rule
            if i != j
                W[i,j] = (1/P)*sum([memories[mu,i]*memories[mu,j] for mu in 1:P])
            end
        end
    end
    
    # Return the trained W matrix
    return W
end



function update_state(state::Vector{Float64}, overlaps::Vector{Float64}, patterns::Matrix{Float64})
    # Updates the state vector based on overlaps and patterns.
    
    # Get the dimension of the state vector
    N = length(state)
    
    # Get the number of overlaps
    P = length(overlaps)
    
    # Initialize a new state vector
    new_state = zeros(N)
    
    # Iterate over dimensions of the state vector
    for i in 1:N
        
        h_i = 0
        # Calculate the weighted sum of inputs for each dimension
        for mu in 1:P
            h_i += patterns[mu,i] * N * overlaps[mu] - P * state[i]
        end
        
        # Update the state based on the sign of the weighted sum
        new_state[i] = sign(h_i)
    end
    
    # Return the updated state vector
    return new_state
end


function overlap(mem_1::Vector{Float64}, mem_2::Vector{Float64})
    # Computes the overlap between two memory vectors.
    
    # Get the lengths of the memory vectors
    N_1 = length(mem_1)
    N_2 = length(mem_2)
    
    # Ensure the memory vectors have the same size
    @assert N_1 == N_2 "Memories sizes don't match"
    
    # Compute the overlap between the memory vectors
    overlap_value = (1 / N_1) * sum([mem_1[i] * mem_2[i] for i in 1:N_1])
    
    return overlap_value
end

function gen_random_memory(size::Int64)
    # Generates a random memory vector of a given size.
    
    # Initialize an empty memory vector
    mem = zeros(size)
    
    # Iterate over the elements of the memory vector
    for i in 1:size
        
        # Generate a random number
        p = rand()
        
        # Assign -1 or 1 based on the random number
        if p < 0.5
            mem[i] = -1
        else
            mem[i] = 1
        end
    end
    return mem
end

function noise_memory(p::Float64, memory::Vector{Float64})
    # Adds noise to a given memory vector.
    
    # Initialize a new memory vector
    new_memory = Float64[]
    
    # Iterate over the elements of the original memory vector
    for (i, s_i) in enumerate(memory)
        
        # Generate a random number
        r = rand()
        
        # Add noise based on the probability p
        if r < p
            push!(new_memory, -1 * s_i)
        else
            push!(new_memory, s_i)
        end
    end
    
    return new_memory
end


########################################################################################

# Parse command line arguments
N = parse(Int64, ARGS[1])
P = parse(Int64, ARGS[2])
N_iter =  parse(Int64, ARGS[3])
p_noise = parse(Float64, ARGS[4])
folder = ARGS[5]

# Initialize matrices for storing memories
initial_memories = zeros(P, N)
state_memories = zeros(P, N)

# Generate initial memories and add noise
for mu in 1:P
    initial_memories[mu, :] = gen_random_memory(N)
end

for mu in 1:P
    state_memories[mu, :] = noise_memory(p_noise, initial_memories[mu, :])
end

# Train W matrix using initial memories
W = w_matrix(initial_memories)

# Initialize a dictionary to store overlaps
overlap_dict = Dict()

# Initialize overlap dictionary with empty arrays
for mu in 1:P
    overlap_dict[mu] = Float64[]
end

##############################################################

# Iterate over time steps
for i in 1:N_iter
    
    # Iterate over each memory
    for mu in 1:P

        # Get the current state of the memory
        mem_mu = state_memories[mu, :]

        # Initialize array to store overlaps with other memories
        overlaps = Float64[]

        # Compute overlap of current memory with all other memories
        for nu in 1:P
            mem_nu = initial_memories[nu, :]
            ovlp_mu_nu = overlap(mem_mu, mem_nu)
            push!(overlaps, ovlp_mu_nu)
        end

        # Update the state of the memory using W matrix
        state_memories[mu, :] = sign.(W * mem_mu)
        
        # Store the maximum overlaps for the last 20 iterations
        if i > (N_iter - 20)
            ovl_max = maximum(overlaps)
            push!(overlap_dict[mu], ovl_max)
        end
        
    end
end

#######################################################################

# Write overlap dictionary to a JSON file
f = open("$(folder)/overlap_$(N)_$(P).json", "w")
JSON.print(f, overlap_dict) 
close(f) 
