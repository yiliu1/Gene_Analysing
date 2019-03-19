#This R scripts is for preprocessing.
#It convert text files to matrix ready for machine learning algorithms.
#By Liu Yi
#2019.02.19
getwd()
setwd("./chow_data")
files <- dir(path="/Users/yi/Desktop/Gene_expression/chow_data",pattern="*\\.txt$")
library(limma)
x <-read.maimages(files,source="agilent",green.only=TRUE,other.columns="gIsWellAboveBG")
library(MmAgilentDesign026655.db)
x$genes$EntrezID <- mapIds(MmAgilentDesign026655.db, x$genes$ProbeName,
                                   keytype="PROBEID", column="ENTREZID")
x$genes$Symbol <- mapIds(MmAgilentDesign026655.db, x$genes$ProbeName,
                                            keytype="PROBEID", column="SYMBOL")
x$genes[201:205,]# to see examples of gene symbols
y <- backgroundCorrect(x, method="normexp")
y <- normalizeBetweenArrays(y, method="quantile")
Control <- y$genes$ControlType==1L
NoSymbol <- is.na(y$genes$Symbol)
IsExpr <-rowSums(y$other$gIsWellAboveBG)>=5
yfilt <- y[!Control & !NoSymbol & IsExpr, ]
yfilt$genes <- yfilt$genes[,c("ProbeName","Symbol","EntrezID")]
gene_data<-yfilt
gene_data$genes<-gene_data$genes[,c("Symbol")]
write.csv(gene_data,file = "gene_data.txt")