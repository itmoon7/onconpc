# Install required packages
#install.packages("reshape2")
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("BSgenome")
#BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
#BiocManager::install("GenomeInfoDb")
#BiocManager::install("deconstructSigs")

library(deconstructSigs)

# Set the data source variable
data_source <- "GENIE"  # Change this to either "DFCI" or "GENIE" based on your data source

# Read mutation data based on data source
if (data_source == "DFCI") {
  setwd("~/Documents/github/onconpc/data/seq_panel_data/dfci/")
  mutationData <- read.csv('profile_mutation_dfci', header = TRUE, sep="\t")
} else if (data_source == "GENIE") {
  mutationData <- read.csv('data_mutations_extended_5.0-public.txt', sep="\t",
                           comment.char="#", header=TRUE, fill=TRUE)
  mutationData <- subset(mutationData, Center %in% c('DFCI', 'MSK', 'VICC'))
}

# Add "chr" prefix to each chromosome name
if (data_source == "GENIE") {
  mutationData[,"CHROMOSOME"] <- paste0("chr", mutationData[,"Chromosome"])
}
# Filter out specific chromosomes
mutationData <- mutationData[!grepl("GL|chrMT", mutationData[,"CHROMOSOME"]), ]
unique(mutationData[,"CHROMOSOME"])

# Convert to deconstructSigs input format based on data source
head(mutationData)
if (data_source == "DFCI") {
  sigs.input <- mut.to.sigs.input(mut.ref = mutationData, 
                                  sample.id = "UNIQUE_SAMPLE_ID", # type: str
                                  chr = "CHROMOSOME", 
                                  pos = "POSITION", 
                                  ref = "REF_ALLELE", 
                                  alt = "ALT_ALLELE")
} else if (data_source == "GENIE") {
  sigs.input <- mut.to.sigs.input(mut.ref = mutationData, 
                                  sample.id = "Tumor_Sample_Barcode", # type: str
                                  chr = "CHROMOSOME", 
                                  pos = "Start_Position", 
                                  ref = "Reference_Allele", 
                                  alt = "Tumor_Seq_Allele2")
}

# Filter samples with low mutations
sigs.input <- sigs.input[rowSums(sigs.input) >= 1,]

# Set filename variable for saving output
if (data_source == "DFCI") {
  filename <- "trinucs_dfci_profile.csv" # Modify this as needed
} else if (data_source == "GENIE") {
  filename <- "trinucs_genie.csv"
}
# Save the output
write.csv(sigs.input, file = filename)

# ===================== [OPTIONAL] =====================
# Visualization of trinucs
trinucs_selected <- sigs.input
results <- vector("list", nrow(trinucs_selected))
names(results) <- row.names(trinucs_selected)

# Select the first 10 samples
subset_data <- head(trinucs_selected, 10)
# run the estimation of exposures for each sample and save the results in the list
for( sID in row.names(subset_data) ){
  results[[sID]] <- whichSignatures(subset_data, # the matrix generated with mut.to.sigs.input 
                                    sample.id=sID, # the current sample ID
                                    signatures.ref=signatures.cosmic, # the data.frame with the signatures that comes with deconstructSigs
                                    tri.counts.method="exome2genome", # which normalization method to use
                                    contexts.needed=TRUE) # set to TRUE if your input matrix contains counts instead of frequencies
}

plotSignatures(results[[1]])


