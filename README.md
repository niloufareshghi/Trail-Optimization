# Trail-Optimization
Provincial Park Trail Development: Optimizing a New Trail in a Canadian Mountainous Region

This project aimed to design a beginner-friendly trail that offers a rewarding experience while minimizing physical exertion for participants. To achieve this goal, an altitude map and measurements from a sports science lab were given to develop a solution that optimizes the trail's design and minimizes expected exertion.

## Ingestion:

To begin the project, I read and processed two CSV files: an altitude map and measurements from the sports science lab. The altitude map had a resolution of 10m × 10m, and it provided crucial information about the terrain. The measurements from the sports science lab established the relationship between walking gradient/slope and the energy expended by multiple test subjects.

## Modeling:

Using the available data, I applied a polynomial regression model to predict a person's expected energy expenditure for a given gradient. By analyzing the relationship between walking gradient/slope and energy expenditure, I developed a model that could estimate the exertion level based on the trail's characteristics. This modeling step allowed us to quantify the expected exertion for different sections of the trail.

![image](https://github.com/niloufareshghi/Trail-Optimization/assets/47944007/16721ce6-6e1e-449d-b9a5-222154316941)


## Optimization:

In the optimization phase, the objective was to find a path from any point on the Southern border of the map to a lookout point at x = 200 and y = 559 that minimizes the total expected exertion in Joules. I utilized the Dijkstra algorithm to find the shortest path from any Southern border point to the target.

## Simple Reporting:

a. Path Solution: This .csv file contains the coordinates of each point along the path, from the starting point on the Southern border to the lookout point at x = 200 and y = 559.

b. Path Instructions: The path solution is displayed as a series of instructions that could guide someone through the trail. 

c. Path Visualization: To enhance the understanding of the trail's design, I overlaid the path on the altitude map, this visualization provided a visual representation of the optimal path, considering the terrain's altitude variations and the trail's alignment.

![path](https://github.com/niloufareshghi/Trail-Optimization/assets/47944007/8fd67ccf-624b-4080-b1d9-54d889adab2f)
