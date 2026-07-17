#          ************************************************************
#          *       Simulation Based Inference - Project  work         *
#          *    Inference of protein secondary structure motifs       *
#          *                        05. July 2026                     *
#          ************************************************************


# Amino acids
aa <- c("A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", 
        "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")

# Alpha-helix emission probabilities
emit_alpha <- c(0.12, 0.06, 0.03, 0.05, 0.01, 0.09, 0.05, 0.04, 0.02, 0.07, 
                0.12, 0.06, 0.03, 0.04, 0.02, 0.05, 0.04, 0.01, 0.03, 0.06)

# Other emission probabilities
emit_other <- c(0.06, 0.05, 0.05, 0.06, 0.02, 0.05, 0.03, 0.09, 0.03, 0.05, 
                0.08, 0.06, 0.02, 0.04, 0.06, 0.07, 0.06, 0.01, 0.04, 0.07)

states <- c("other",  "alpha")

# Transition matrix
transition <- matrix(c(0.95, 0.05,   # other -> other / alpha
                       0.10, 0.90),  # alpha -> other / alpha
                     byrow = TRUE, nrow = 2)

rownames(transition) <- states
colnames(transition) <- states

transition


# ---------------------------------------------------------
# Simulates the hidden state sequence of the HMM
#
# Parameters: n - Length of the sequence.
#             transition - Transition probability matrix.
#
# Returns: A character vector containing the hidden states
#          ("other" or "alpha") for each sequence position
# ---------------------------------------------------------

simulate_states <- function(n, transition){
  
  # Get the names of all possible hidden states
  states <- rownames(transition)
  
  # Allocate memory for the hidden state sequence
  hidden <- character(n)
  
  # According to the assignment, every sequence starts
  # in the "other" state.
  hidden[1] <- "other"
  
  # Generate the remaining hidden states
  for(i in 2:n){
    
    # Current hidden state
    current <- hidden[i-1]
    
    # Sample the next state according to the transition
    # probabilities of the current state.
    hidden[i] <- sample(states, size = 1, prob = transition[current, ])
  }
  return(hidden)
}

# ---------------------------------------------------------
# Generates an amino acid sequence based on the hidden
# state sequence
#
# Parameters: hidden - Character vector of hidden states
#
# Returns: Character vector containing one amino acid
#          for each hidden state
# ---------------------------------------------------------

simulate_sequence <- function(hidden){
  
  # Allocate memory for the amino acid sequence
  sequence <- character(length(hidden))
  
  # Generate one amino acid for every hidden state
  for(i in seq_along(hidden)){
    
    if(hidden[i] == "alpha"){
      
      # Sample from the alpha-helix emission distribution
      sequence[i] <- sample(aa, size = 1, prob = emit_alpha)
    } else {
      # Sample from the "other" emission distribution
      sequence[i] <- sample(aa, size = 1, prob = emit_other)
    }
  }
  return(sequence)
}

# ---------------------------------------------------------
# Simulates a complete Hidden Markov Model (HMM)
#
# The function first generates the hidden secondary
# structure states and then emits an amino acid for each
# hidden state
#
# Parameter: n   Length of the protein sequence
#
# Returns: Data frame containing:
#           - Position
#           - Hidden state
#           - Simulated amino acid
# ---------------------------------------------------------

simulate_hmm <- function(n){
  
  # Simulate the hidden state sequence
  hidden <- simulate_states(n, transition)
  
  # Generate amino acids from the hidden states
  sequence <- simulate_sequence(hidden)
  
  # Combine everything into a single data frame
  data.frame(Position = 1:n, State = hidden, AminoAcid = sequence)
}


set.seed(1)

protein <- simulate_hmm(30)
protein

# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

# Test 1: Initial state
protein$State[1]


# Test 2: Transitions
set.seed(1)

hidden <- simulate_states(100000, transition)

table(hidden)

from_other <- hidden[-length(hidden)] == "other"
to_alpha   <- hidden[-1] == "alpha"

mean(to_alpha[from_other])


from_alpha <- hidden[-length(hidden)] == "alpha"
to_alpha   <- hidden[-1] == "alpha"

mean(to_alpha[from_alpha])

# Test 3: Emissions
set.seed(1)

seq_alpha <- simulate_sequence(rep("alpha", 100000))

prop.table(table(seq_alpha))

seq_other <- simulate_sequence(rep("other", 100000))

prop.table(table(seq_other))
