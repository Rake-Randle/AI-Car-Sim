# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)
# This code has again been hoisted by the CGS Digital Innovation Department
# giving credit to the above authors for the benfit of our education in ML

#GLOBAL VARIABLE

import math
import random
import sys
import os

import neat
import pygame

# Constants
WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 50
CAR_SIZE_Y = 50

BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit

current_generation = 0  # Generation counter
"""
The Car Class 

Throughout this section, you will need to explore each function
and provide extenive comments in the spaces denoted by the 
triple quotes(block quotes) """ """.
Your comments should describe each function and what it is doing, 
why it is necessary and where it is being used in the rest of the program.

"""


class Car:
    """1. This Function:
    This Function defines the "self" which is the car's properties and values. This function defines properties such as, how the car looks (png file), 
    scales the car so it fits with the pygame window, rotates the image. That just defines how it looks, but this function also defines it's,
    starting position, the angle of which it faces, the starting speed (Later modified in other functions), calculates the center of the sprite,
    creates all the radars and sensors to be drawn. The function also defines the car as crashed when it starts so it resets the map for the begining.
    This function is very significant for the start and the some of the global variables that will be used throughout the code.
    """

    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load( 
            "car3.png"
        ).convert()  # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.position = [1725, 840]  # Starting Position
        self.angle = 90 # Starting Angle
        self.speed = 0

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [
            self.position[0] + CAR_SIZE_X / 2,
            self.position[1] + CAR_SIZE_Y / 2,
        ]  # Calculate Center

        self.radars = []  # List For Sensors / Radars
        self.drawing_radars = []  # Radars To Be Drawn

        self.alive = True  # Boolean To Check If Car is Crashed

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed

    """ 2. This Function:
        has two parameters, self and screen (later defined) which is then used to draw and add the car and the sensor
        onto the screen so it is visual for everyone to see. The '.blit' in this function draws the car from the last function.
    """

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)  # Draw Sprite
        self.draw_radar(screen)  # OPTIONAL FOR SENSORS

    """ 3. This Function:
        Is used to draw all the sensor and radars onto the pygame window. Within in the for loop, it draws the lines
        and the circle points for the cars sensors, this helps visualise the radar and sensor in the simulation. 
        In this function you can define the colour and the thickness of the radars.  
    """

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (44,111,187), self.center, position, 2)
            pygame.draw.circle(screen, (44,111,187), position, 7)

    """ 4. This Function:
        This function checks if the car sprite has collided with any wall and if it does it will kill the sprite. 
        It first defines the car as being alive and then has a for loop to continue to check if the radius and the border of the car has hit a border
        The different points create a rectangle/ square around the car, and then checks if it has hit the border colour, which is white.
        If the square/ rectangle hits the border colour it will kill the car and break the function.
    """

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    """ 5. This Function:
        This function is essential for the learning of the car. The function calculates the distance from the cars
        to the obstacles in specific directions. These measurments are then pushed to the radar readings which 
        allow the car to load. The function also checks while the radars aren't hitting the border then it extends
        the radars to its max length. All the calculations of distances are then pushed towards an array for data
        usage in the future which enables the car to learn for future generations. 
    """

    def check_radar(self, degree, game_map):
        length = 50
        x = int(
            self.center[0]
            + math.cos(math.radians(360 - (self.angle + degree))) * length
        )
        y = int(
            self.center[1]
            + math.sin(math.radians(360 - (self.angle + degree))) * length
        )

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 350:
            length = length + 1
            x = int(
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + degree))) * length
            )
            y = int(
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + degree))) * length
            )

        # Calculate Distance To Border And Append To Radars List
        dist = int(
            math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2))
        )
        self.radars.append([(x, y), dist])

    """ 6. This Function:
        Updates the car's position, angle, and radar based off of its current actions/ state. 

        The 'if not self.speed_set:' condition sets the car's inital speed to 20 if the 
        'self.speed_set' is set to false, and then it is set to true so that the speed won't go back to the starting speed.

        This fucntion also rotates the sprite to match the starting angle. It also calulates (using trigonometry) to change the 
        X and Y positions so it doesn't crash at the start, this also gives the cars room by calculating the width. 
        The same process of calculations using trigonometry is used in both the X and Y axis for better performance.

        The function increases the distance and the time of the cars so it can track the perfoamce of the cars
        which acts as the rewards and fitness for the machine learning aspect of the code. 

        The below code calculates the new centre of the car by calculating the half the width and height of the car.
        
        The function calculates the bounding box around the car based on the car's angle, dimensions, and centre. 
        These points create the cars collision points. 

        The function continuously checks if the car has collided with any border based off the car's postion.
        It also continuosly picks up data off the radars which are placed every 45 degrees to gather the different obstacles. 
    """

    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [
            int(self.position[0]) + CAR_SIZE_X / 2,
            int(self.position[1]) + CAR_SIZE_Y / 2,
        ]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length,
        ]
        right_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length,
        ]
        left_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length,
        ]
        right_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length,
        ]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    """ 7. This Function:
        This function gets the data off the radars which calculates the distance between the car and the border. 
    """

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    """ 8. This Function:
    
    """

    def is_alive(self):
        # Basic Alive Function
        return self.alive

    """ 9. This Function:
    
    """

    def get_reward(self):
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        return self.distance / (CAR_SIZE_X / 2)

    """ 10. This Function:
    
    """

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image


""" This Function:

1.  Empty Collections Initialization:
    nets: A list to hold neural networks corresponding to each genome.
    cars: A list to hold instances of the Car class that the neural networks control.
    
2.  Initializing PyGame and Display:
    The Pygame library is initialized, and a display window is created with the specified dimensions.

3.  Creating Neural Networks and Cars:
    For each genome passed into the run_simulation function:
    A neural network is created using the neat.nn.FeedForwardNetwork.create(g, config) method. The genome g and configuration config are used to create the network.
    The neural network is added to the nets list, and the genome's fitness is set to 0.
    A new instance of the Car class is created and added to the cars list.

4.  Clock and Font Settings:
    A PyGame clock is created to control the frame rate of the simulation.
    Different fonts are loaded for displaying information on the screen.

Updating the Generation Counter:

The current_generation global variable is incremented, indicating the current generation being simulated.
Main Simulation Loop:

The loop runs indefinitely, simulating the behavior of each car and updating the neural networks based on their actions.
Event Handling:

PyGame events are processed within the loop.
If the user closes the window or presses the escape key, the program exits.
Car Actions and Neural Network Activation:

For each car, the neural network is activated with the car's sensor data using nets[i].activate(car.get_data()).
The highest value in the output of the neural network determines the car's action: left, right, slow down, or speed up.
Updating Fitness and Car Movement:

For each alive car, the car's fitness is increased, and its position and movement are updated based on its action.
Checking Car Survival:

The number of cars that are still alive is counted.
If no cars are alive, the simulation loop is terminated.
Time Limit for Simulation:

A simple counter is used to roughly limit the duration of the simulation.
Drawing the Game Environment:

The game map is drawn on the screen.
For each alive car, its image is drawn on the screen.
Displaying Information:

Text information about the current generation, number of cars alive, and mean fitness is displayed on the screen.
Updating Display and Frame Rate:

The display is updated with the drawn elements.
The frame rate is controlled using the clock, ensuring a maximum of 60 frames per second.


"""


def run_simulation(genomes, config):
    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    title_font = pygame.font.SysFont("Arial", 35)
    game_map = pygame.image.load("map.png").convert()  # Convert Speeds Up A Lot
    
    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0

    while True:
        # Exit On Quit Event
        """
        Mod: added on keydown/esc to quit the game
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit(0)

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                if car.speed - 2 >= 12:
                    car.speed -= 2  # Slow Down
            else:
                car.speed += 2  # Speed Up

        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:  # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Display Info
        text = title_font.render("Mt Panorama", True, (227,91,137))
        text_rect = text.get_rect()
        text_rect.center = (590, 255)
        screen.blit(text, text_rect)

        text = generation_font.render(
            "Generation: " + str(current_generation), True, (227,91,137)
        )
        text_rect = text.get_rect()
        text_rect.center = (590, 285)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (227,91,137))
        text_rect = text.get_rect()
        text_rect.center = (590, 315)
        screen.blit(text, text_rect)

        text = alive_font.render(
            "Time: " + str(round(counter/60)), True, (227,91,137)
        )
        text_rect = text.get_rect()
        text_rect.center = (590, 345)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS


""" 1. This Section: The program main section
    The if __name__ == "__main__": block ensures that the code within it 
    only executes when the script is run directly (not when imported as a module).
    
    The config.txt file settings are loaded into the config variable using Config()
        neat.DefaultGenome
            Various options that control genome node activation, aggregation, bias, 
            compatibility, connection management, feed-forward architecture, response, 
            and weight settings.
            num_hidden, num_inputs, num_outputs: 
            Specifies the number of hidden, input, and output nodes, respectively.
            These parameters collectively define the structure and characteristics 
            of the neural networks represented by the genomes.
            
        neat.DefaultReproduction
            elitism: Specifies the number of elite genomes that are directly passed 
            to the next generation.
            survival_threshold: Sets the survival threshold, indicating the proportion 
            of genomes in each species that are considered for reproduction.
            
        neat.DefaultSpeciesSet
            compatibility_threshold: Specifies the compatibility threshold used for 
            determining species separation. Genomes with compatibility distance below 
            this threshold belong to the same species.
            
        neat.DefaultStagnation
            species_fitness_func: Defines the function to use when determining species 
            fitness. In this case, 'max' indicates that the maximum fitness of a 
            species is used.
            max_stagnation: Specifies the maximum number of generations a species 
            can remain stagnant before it's considered for stagnation and possible 
            extinction.
            species_elitism: Specifies the number of elite genomes from each species 
            that are preserved to the next generation.
            
        config_path
    
    
"""
if __name__ == "__main__":
    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run Simulation For A Maximum of 1000 Generations
    population.run(run_simulation, 1000)
