

X = [LinRange(0,1,100)|>collect; LinRange(5,6,100)|>collect]
U,V = aca2((x,y)->1/(1+(x-y)^2), X, X, 20)
A = FullMat((x,y)->1/(1+(x-y)^2), X, X)
maximum(abs.(A-U*V'))
# pcolor(abs.(A-U*V'))
# colorbar()