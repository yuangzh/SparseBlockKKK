function [A,b] = getdata(iwhich)


switch iwhich
    
    case 1
        n = 1024;
        totnnz = 100;
        x = generate_x(n,totnnz);
        A = randn(256,n);
        b = A*x + generate_noise(256,1);

    case 2
        n = 2048;
        totnnz = 100;
        x = generate_x(n,totnnz);
        A = randn(256,n);
        b = A*x + generate_noise(256,1);
        
    case 3
        n = 1024;
        load E2006_5000_10000;
        [data_m,data_n]=size(x);
        seq = randperm(data_n,n);
        A = x(:,seq); b = y;
        
    case 4
        n = 2048;
        load E2006_5000_10000;
        [data_m,data_n]=size(x);
        seq = randperm(data_n,n);
        A = x(:,seq); b = y;
    
%     case 5
%         load YearPredictionMSD_5000;
%         A = x; b = y;
        
        
%%%%%%%%%%%%%%%        
    case 11
        n = 1024;
        totnnz = 100;
        x = generate_x(n,totnnz);
        A = randn(256,n);A = scaleA(A);
        b = A*x + generate_noise(256,2);


    case 12
        n = 2048;
        totnnz = 100;
        x = generate_x(n,totnnz);
        A = randn(256,n);A = scaleA(A);
        b = A*x + generate_noise(256,2);

        
    case 13
        n = 1024;
        load E2006_5000_10000;
        [data_m,data_n]=size(x);
        seq = randperm(data_n,n);
        A = x(:,seq); A = scaleA(A); b = y;
        
    case 14
        n = 2048;
        load E2006_5000_10000;
        [data_m,data_n]=size(x);
        seq = randperm(data_n,n);
        A = x(:,seq); A = scaleA(A); b = y;
    
%     case 15
%         load YearPredictionMSD_5000;
%         A = x; A = scaleA(A); b = y;
        
        
        

        
        
        
 
       
end



