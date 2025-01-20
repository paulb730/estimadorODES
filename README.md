### Features

- Support 5 types of heuristic algorithms PSO GWO BE DE GS
- Full-featured: Data modeling visualization, model description variables, parameters and ODES
- Support parameter algorithms manipulation
- Support graphic data tools and image downloading
- Compatible with all major browsers (IE8+), compatible with mobile devices
# ODES Parameter Estimating

![](https://github.com/paulb730/estimadorODES/blob/main/Banner.png?raw=true)

**Table of Contents**

[TOCM]

[TOC]

#Installation
###Requirements
- Python 3.8.12
- Conda 4.8.3
- CUDA  12.6.85


####Clone Repository

`$ git clone https://github.com/paulb730/estimadorODES.git `

#### Create a conda environment 

Open CMD or ANACONDA PROMPT

    $ cd estimadorODES
	$ cd src
	$ conda create --name <env> --requirements.txt 
    $ conda activate --envcreated

Execute App:

    python plotly_interface.py
##Abstract
The present project proposes to develop a web application that allows the
execution of algorithms based on collective intelligence. It means that through the management of libraries and frameworks related to data visualization and analysis are done. It has the objective to keep record and compare the performance of PSO with other similar algorithms in the estimation of parameters of ordinary differential equations. This project will analyze case studies with real experimental data through bibliographic research and will execute the proposed algorithms to simulate data obtained from a mathematical model. In order to establish statistics regarding the performance and precision of the parameters resulting from the estimation
process.
##Study Cases
####Enzimatic Model
The main goal to get modeled a enzimatic problem is measure blood enzymes rate and analyze how these contribuite in a myocardial infarction.
####VHC Dynamic
The mathematical model establishes 4 populations (healthy and diseased hepatocytes, viral load and cytotoxic T cells), taking measurements of viral load and cytotoxic T cells.
The main goal of the model is to predict the status of an individual
infected by the hepatitis C virus and to monitor the evolution of its viral load and liver damage, without the need for biopsies (Alavez Ramirez, 2007).
####Benchmark
Test model to measure performance of each algorithms as like different set of parameters.
####Lotka Volterra
This model represents three different competitors in the mobile telecommunication systems market in Greece. The main goal is to predict the possibility of ensuring a future stable and healthy competitive equilibrium in the market (Kloppers & Greeff, 2013).
####HIV Dynamic
HIV virus dynamics is a proposed mathematical model that aims to analyze the viral load of a person over time and thus estimate parameters that allow us to quantify and relate the response of the immune system, the virus in the body, and the application of treatments (Cruz-Langarica, Valle-Trujillo, Rios, SoteloOrozco, & Plata-Ante, 2017).
####Chemical Kinetic
Kinetics in chemical reactions is the study of the rate at which chemical reactions take place. The results can only be achieved experimentally and from them, the pathway through which these reactions take place can be predicted (Chena, Aguilar,
& Cano-Estepa, 2009).
## User GUIDE
- Select a study case
![](https://raw.githubusercontent.com/paulb730/estimadorODES/86f06d211199cb09b950f306a7192823864f067d/1st.png)

- Check model description and experimental data 
![](https://raw.githubusercontent.com/paulb730/estimadorODES/86f06d211199cb09b950f306a7192823864f067d/2.png)

- Choose and heuristic algorithm
![](https://raw.githubusercontent.com/paulb730/estimadorODES/86f06d211199cb09b950f306a7192823864f067d/3.png)

- Check functions of the application as Integration
![](https://raw.githubusercontent.com/paulb730/estimadorODES/86f06d211199cb09b950f306a7192823864f067d/4.png)

- Execute Parameter Estimation
![](https://raw.githubusercontent.com/paulb730/estimadorODES/86f06d211199cb09b950f306a7192823864f067d/5.png)

- Check results
![](https://raw.githubusercontent.com/paulb730/estimadorODES/86f06d211199cb09b950f306a7192823864f067d/67.png)

###End
