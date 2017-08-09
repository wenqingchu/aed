function  Re = recError(X, R, ThrTest) 
   for ii = 1 : size(X,2)
       X(:,ii) = X(:,ii)/norm(X(:,ii),2);
   end
   K =  length(R);
   reSet = 1 : size(X,2);
   Re = ones(1,size(X,2));
   for ii = 1 : K 
       Re(reSet) = sum((R(ii).val*X(:,reSet)).^2); 
       idx = find(Re < ThrTest);
       reSet = setdiff(reSet,idx);
   end
 
   
%    K =  length(R);
%    Re = ones(1,size(X,2));
%    tmp = ones(K, size(X,2));
%    for ii = 1 : K 
%        Re = sum((R(ii).val*X(:,:)).^2);
%        tmp(ii,:) = Re;
%    end
%    Re = min(tmp);
%        

end