library(tidyverse)
library(readxl)

data <- read_xlsx("benchmark_hpc_100000.xlsx")

data <- data[-11,]

data$Method <- factor(data$Method, levels = c("f32CPU_SV",
                                              "i32CPU_SV",
                                              "f32CPU_DV",
                                              "i32CPU_DV",
                                              "f32CPU_SM",
                                              "i32CPU_SM",
                                              "f32CPU_DM",
                                              "i32CPU_DM",
                                              "f32GPU_DV",
                                              "f32GPU_DM",
                                              "f32GPU_SM"))

std9_palette = c("#2B2D42", "#5C6378", "#8D99AE",
                 "#BDC6D1", "#EDF2F4", "#EE8B98",
                 "#EF233C", "#E41433", "#D90429")

grp9_palette = c("#2B2D42", "#5C6378", "#757E93",
                 "#A5B0C0", "#BDC6D1", "#EDF2F4",
                 "#EF233C", "#E41433", "#D90429")

nge9_palette = c("#E8D5D3", "#CDC1C4", "#B2ADB4", 
                 "#C06B6E", "#CE2928", "#B62626", 
                 "#9E2224", "#6A2123", "#362022")

grp11_palette = c("#2B2D42", "#383B50", "#44485D", "#5C6378",
                  "#BDC6D1", "#D5DCE3", "#E1E7EC", "#EDF2F4",
                  "#EF233C", "#E41433", "#D90429")

grp10_palette = c("#2B2D42", "#383B50", "#44485D", "#5C6378",
                  "#BDC6D1", "#D5DCE3", "#E1E7EC", "#EDF2F4",
                  "#EF233C", "#E41433")

text_col <- c(rep("white", 4), rep("black", 4), rep("white", 2))

# export as 1200 x 800

ggplot(data, aes(x = Method, y = Mean, fill = Method)) +
  geom_bar(stat="identity", color = "black") +
  geom_errorbar(aes(x = Method, ymin = Mean-SD, ymax = Mean+SD), colour = "black", width = 0.3, linewidth = 1.0) +
  scale_fill_manual(values = grp10_palette) +
  ggtitle("Execution time of the different methods using 100 000 simulated candidates.") +
  xlab("Method") +
  ylab("Time (s)") +
  geom_text(aes(label=round(Mean, 2)), size = 5.0, color = text_col, position = position_stack(vjust = 0.5)) +
  theme_minimal(base_size = 18) +
  theme(axis.text.x = element_text(angle = 90, hjust = 0, vjust = 0.5))
