# Simulate a city
Welcome to the repository of group 8

## Project
The model we build is a simulation of a city. How the city grows depending on multiple parameters and rules.

## Activities
We make use of three different activities and streets.
We have houses, which generate other houses, industry and even stores.
We have industry, which can only spawn industry and stores which can only spawn stores.
Streets can only be build when there is a specific threshold of activity in the neighbourhood, and houses and industry can only be build when streets are present.
This way, the development of the city follows the development of the streets and the other way around.

## Parameters
Our model makes use of multiple parameters. Considering there are a lot of restrictions, the parameters need to be tuned in order to get a fully developed, heterogenous city.

## Papers
We made use of a paper by Batty, which can be found [here](https://www.ucl.ac.uk/bartlett/casa/sites/bartlett/files/ceus-paper.pdf).

## Website
Please click [here](https://compsysgroupeight.wordpress.com/) to go to our website. Here you can find more information and see more simulations.

## Simulation
![Here is a random simulation of our model](https://github.com/RoelvdBurght/complex-systems/blob/master/RandomPicture.png)

We see that it contains streets (grey), houses (green), industry (yellow) and stores (blue).

## Running the code
Our simulation is run by running our jupyter notebook called *simulate_city.ipynb*.
For changing the parameters, go to *city_class.py* and change them in the corresponding thresholds.

### Contributors
Bart van Laatum

Roel van der Burght

Sami Achetib

Wiebe Jelsma
