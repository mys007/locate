function W = significantEdges(dmap, Wthres)

% extract edges in log domain (but the domain really doesn't matter that much)
dmap(dmap<=0) = 1e-20;
E = edge(log(dmap),'Canny',[1e-7 6e-4]);  %or edge(data,'Canny',[0.0001 0.005])

% weight edges by significance (=how deep is the drop = derivative)
% (we use log-domain because otherwise far edges are weighted too much without being visually striking)
W = imgradient(log(dmap));
W(E==0)=0;

%cull horizon and similar (overly strong) and normalize to [0,1]
W(W>Wthres) = Wthres;
W = W/Wthres;

%figure(2); imagesc(E)
%figure(3);  imagesc(W)

