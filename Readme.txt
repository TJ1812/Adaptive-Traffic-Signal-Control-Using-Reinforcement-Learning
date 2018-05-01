===========================================================================
ADAPTIVE TRAFFIC CONTROL WITH DEEP REINFORCEMENT LEARNING
===========================================================================

Introduction

The rapid increase in automobiles on roadways has naturally lead to traffic congestions all over the world, forcing drivers to sit idly in their cars wasting time and needlessly consuming fuel. Ride sharing and infrastructural improvements can help mitigate this but one of the key components to handling traffic congestion is traffic light timing. Traffic light control policies are often not optimized, leading to cars waiting pointlessly for nonexistent traffic to pass on the crossing road. We feel that traffic light control policy can be greatly improved by implementing machine learning concepts. Our project focuses on implementing a learning algorithm that will allow traffic control devices to study traffic patterns/behaviors for a given intersection and optimize traffic flow by altering stoplight timing. We do this with a Q-Learning technique, similarly seen in previous works such as Gao et. al. and Genders et. al, where  an intersection is knowledgeable of the presence of vehicles and their speed as they approach the intersection. From this information, the intersection is able to learn a set of state and action policies that allow traffic lights to make optimized decisions based on their current state. Our work seeks to alleviate traffic congestion on roads across the world by making intersections more aware of traffic presence and giving them the ability to take appropriate action to optimize traffic flow and minimize waiting time.


Setup

This project was developed using python 3. Install from python.org .

Prerequisite Packages :
1) Tensorflow (pip install tensorflow) or (pip install tensorflow-gpu)
2) Keras (pip install keras)

The simulatons were done on SUMO Traffic Simulator. It is available at following location :- http://sumo.dlr.de/wiki/Downloads

Please refer to installation instructions of sumo from the website.



Running

Run file - traffic_light_control.py












