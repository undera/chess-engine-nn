
# Chess Engine with Neural Network

## Motivation

It is inspired by TCEC Season 14 - Superfinal, where Leela was trying to fry Stockfish. [Stockfish](https://stockfishchess.org/) is chess engine that dominates among publicly available engines since 2014. Stockfish uses classic approach of chess engines, where everything is determined by algorithms built into engine. [Lc0 aka Leela Chess Zero](http://lczero.org/) is different, since it uses some sort of trainable neural network inside, as well as appealing human-like nickname.

I wanted to demonstrate my 7yr-old daughter that making these computer chess players are not as hard. 

## Journal

### Feb 15, 2019
First commits of the code.
Using own representation of chess board as 8 * 8 * 12 array of 1/0 values. 1/0 are used as most distinctive input for NN about piece presence at certain square. NN uses two hidden layers 64 nodes each. Output of NN is 8 * 8 array for "from cell" and 8 * 8 array for "to cell" scores. A piece of code is used to choose best move that a) is by chess rules; b) has best score of "from" multiplied by "to" output cells. Finally, board state is checked for game end conditions of checkmate or 50-move rule.

Two copies of NN are used, one plays as White and one as Black, playing versus each other. Game is recorded as PGN file, to be able to review it by human.
