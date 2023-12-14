# Install required packages
# install.packages("reshape2")
# if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
# BiocManager::install("BSgenome")
# BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
# BiocManager::install("GenomeInfoDb", force = TRUE)
# BiocManager::install("deconstructSigs", force = TRUE)
import rpy2.robjects as ro
from rpy2.robjects import conversion, default_converter

def get_base_substitutions():
  with conversion.localconverter(default_converter):

    ro.r("""   
        library(deconstructSigs)

          mutationData <- read.csv('./mutation_input.csv')
        
          if(nrow(mutationData) == 0) {
            return("Empty mutation data. No processing done.")
          }
        
          mutationData <- mutationData[!grepl("GL|chrMT", mutationData[,"CHROMOSOME"]), ]
          unique(mutationData[,"CHROMOSOME"])

          # Convert to deconstructSigs input format based on data source
          head(mutationData)
          sigs.input <- mut.to.sigs.input(mut.ref = mutationData, 
                                    sample.id = "UNIQUE_SAMPLE_ID", # type: str
                                    chr = "CHROMOSOME", 
                                    pos = "POSITION", 
                                    ref = "REF_ALLELE", 
                                    alt = "ALT_ALLELE")
    
          # Filter samples with low mutations
          sigs.input <- sigs.input[rowSums(sigs.input) >= 1,]

          # Set filename variable for saving output

          filename <- "trinucs_userinput_profile.csv" # Modify this as needed
    
          # Save the output
          write.csv(sigs.input, file = filename)""")

  return "./trinucs_userinput_profile.csv"

  


