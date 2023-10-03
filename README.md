# commahack4_chatpid


## chatPID: AI-driven Navigation using LLMs

**Objective**: Provide seamless navigation for a robot by leveraging Language Models.

## Video:
[Link to Demo](https://youtu.be/VT-i3yRsX2s?t=3214)
[Link to Longest Run with Code View](https://www.youtube.com/watch?v=duRuXlKctz0&ab_channel=BrianMachado)

### Overview:
chatPID employs a unique combination of image segmentation and natural language processing to determine the optimal navigation path. It takes camera images as input, processes them through a series of steps, and finally communicates with a Language Model to determine the best set of movement commands for a robot.

### Workflow:

#### **1. Image Segmentation using SAM (Segment Anything Model)**
```
[ Camera Image ] ---> [ SAM (Segment Anything Model) ] ---> [ Segmented Image ]
```
The purpose of this step is to segment the raw image into discernible regions.

#### **2. Heuristic Labelling of Segmented Image**
```
[ Segmented Image ] 
      |
      V
[ Heuristic Labeller ] 
      |
      V
[ Labelled Image ]
```
Given that SAM doesn't provide direct labels:
- Regions larger than 10% of the image are considered significant structures like walls or floors.
- A region's edge contact helps in distinguishing between a wall and a floor. If more pixels of a significant region touch the top, left, or right edges, it's labeled as a wall.

#### **3. Bucketing & Averaging**
```
[ Labelled Image ] 
      |
      V
[ Bucketing & Averaging ] 
      |
      V
[ Bucketed Image ]
```
The image is divided into 30x30 buckets. Each bucket is labeled based on the average labels of the regions it contains, functioning somewhat like a convolution operation.

#### **4. ASCII Art Generation**
```
[ Bucketed Image ] 
      |
      V
[ ASCII Generator ] 
      |
      V
[ ASCII Image ]
```
A 2D ASCII array is produced to represent the robot's perspective from the camera. This serves as an abstraction of the environment, simplifying the information that needs to be processed.

#### **5. Anchoring**
```
[ ASCII Image ] 
      |
      V
[ Anchor Adder ] 
      |
      V
[ Anchored ASCII Image ]
```
Key landmarks are injected into the ASCII representation:
- "CURRENT LOCATION" is placed at the bottom center, representing the robot's current position.
- Anchors like "TOP LEFT" and "TOP RIGHT" are added to provide context.

TOP LEFT Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall TOP RIGHT
Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall
Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall
Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall
Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall
Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Floor Floor Floor Floor Wall Wall Wall Wall Wall
Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Floor Floor Floor Floor Wall Wall Wall Wall Wall
Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Floor Floor Floor Floor Floor Floor Wall Wall Wall Wall
Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Floor Floor Floor Floor Floor Floor Wall Wall Wall Wall
Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Floor Floor Floor Floor Floor Floor Wall Wall Wall Wall
Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Floor Floor Floor Floor Floor Floor Floor Floor Wall Wall Wall
Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Wall Floor Floor Floor Floor Floor Floor Floor Floor Wall Wall Wall
BOTTOM LEFT Wall Wall Wall Wall Wall Wall Wall Wall Wall Floor CURRENT LOCATION Floor Floor Floor Floor Floor Floor Floor Floor Floor Wall BOTTOM RIGHT

#### **6. Heuristic Descriptions for Context**
```
[ Anchored ASCII Image ] 
      |
      V
[ Description Generator ] 
      |
      V
[ Descriptive Text ]
```
Before feeding data to GPT-4, the environment is described in natural language to provide a high-level context.

#### **7. Path Planning with GPT-4**
```
[ Descriptive Text + Anchored ASCII Image ] 
                             |
                             V
[ GPT-4 Model for Navigation Decisions ] 
                             |
                             V
[ Navigation Commands ]
```
With all the preprocessed data, GPT-4 is consulted to generate a navigation path in W, A, S, D space.
