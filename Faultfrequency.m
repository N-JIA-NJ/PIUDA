function [L_en] = Faultfrequency(order,label,plabel)
load Datas  
fs=10000;
n=1024;
DATA=Datas; 
% 
f1=35;
f2=63; 
f3=15;
f4=42;

L_en=zeros(1,10);
P=zeros(4,10);

for i=1:length(order)

    B=DATA(:,order(:,i));
    nfft = fs;
    fz=(0:nfft/2)*fs/nfft;
    Y=fft(B,nfft);
    imfFFT=abs(Y);
    YY=imfFFT(1:nfft/2+1)/nfft;

for j=1:3

        f11=floor(f1*j);
        f22=floor(f2*j);
        f33=floor(f3*j);
        f44=floor(f4*j);

        en1(j)=YY(f11+1)^2+YY(f11+2)^2;
        en2(j)=YY(f22+1)^2+YY(f22+2)^2;
        en3(j)=YY(f33+1)^2+YY(f33+2)^2;
        en4(j)=YY(f44+1)^2+YY(f44+2)^2;
        
end
en=[sum(en1),sum(en2),sum(en3),sum(en4)];
ENP1=en/sum(en);
ENP=1./ENP1;
[~,p]=min(ENP);
P(p,i)=1;

 [~,a2]=max(label(:,i));
 [~,b2]=max(plabel(:,i));
 [~,c2]=max(P(:,i));

    if a2==b2
        L_en(1,i)=0;
    end
    if a2~=b2
     if a2==c2
        if b2 == 1
        L_en(1,i)=ENP(1,1);
        end
        if b2 == 2
        L_en(1,i)=ENP(1,2);
        end
        if b2 == 3
        L_en(1,i)=ENP(1,3);
        end
        if b2 == 4
        L_en(1,i)=ENP(1,4);
        end
     end
     if a2~=c2
        [m,~]=max(ENP);
        L_en(1,i)=m;
     end
    end
end
  
