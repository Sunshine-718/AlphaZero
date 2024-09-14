#  Makefile
#  Nine-Board Tic-Tac-Toe
#  COMP3411/9814 Artificial Intelligence
#  Alan Blair, CSE, UNSW

CC = gcc
CFLAGS = -Wall -g -O3

default: agent

agent: agent.o client.o game.o utils.o Negamax.o heuristic.o common.h agent.h game.h utils.h Negamax.h heuristic.h
	$(CC) $(CFLAGS) -o agent agent.o client.o game.o utils.o Negamax.o heuristic.o

servt: servt.o game.o common.h game.h agent.h
	$(CC) $(CFLAGS) -o servt servt.o game.o

all: agent servt clean

%o: %c common.h agent.h Negamax.h heuristic.h utils.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o
