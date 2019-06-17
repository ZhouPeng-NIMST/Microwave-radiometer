clear;close all;
load norain13_17.mat
path_in = 'F:\data\beijing2018\ÃΩø’\Œ¥≤Â÷µ\1317_54511(Œ¥≤Â÷µ).xlsx';
path_out = 'F:\data\beijing2018\ÃΩø’\≤Â÷µ∫Û';
[num,date,~]=xlsread(path_in);
out_1={1};
k=1;
m=1;
for i=1:4:size(num,2)
    %if  ismember(date{1,i}(1:10),raindate2011)
    %disp(date{1,i}(1:10))
    %end
    if  ismember(date{1,i}(1:10),norain1317)%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp (m);
        m=m+1;
        PRES=num(1:end,i);
        PRES=PRES(~any(isnan(PRES),2),:);
        if length(PRES) < 30
            continue
        end
        HGHT=num(1:end,i+1);
        HGHT(1) = 0;
        TEMP=num(1:end,i+2);
        RELH=num(1:end,i+3);
        any(~isnan(HGHT));
        HGHT=HGHT(~any(isnan(HGHT),2),:);
        TEMP=TEMP(~any(isnan(TEMP),2),:);
        RELH=RELH(~any(isnan(RELH),2),:);
        HGHT_1=[0,0.01,0.025,0.05,0.075,0.1,0.13,0.16,0.19,0.22,0.25,0.28,0.31,0.34,0.37,0.4,0.43,0.46,0.49,0.52,0.56,0.60,0.64,0.68,0.72,0.76,0.8,0.84,0.88,0.92,0.96,1,1.04,1.08,1.12,1.16,1.2,1.26,1.32,1.38,1.44,1.50,1.56,1.62,1.68,1.74,1.8,1.89,1.98,2.17,2.26,2.35,2.43,2.50,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.65,3.8,3.95,4.1,4.25,4.4,4.55,4.6,4.8,5,5.2,5.4,5,6,5.8,6,6.3,6.6,6.9,7.2,7.5,7.8,8.1,8.4,8.7,9,9.3,9.8,10];
        PRES_1=abs(interp1(HGHT,PRES,HGHT_1*1000,'spline','extrap'));
        PRES_1=PRES_1';
        if isnan(PRES_1(end))
            continue
        end
        RELH_1=abs(interp1(HGHT,RELH,HGHT_1*1000,'pchip','extrap'));
        RELH_1=RELH_1';
        TEMP_1=interp1(HGHT,TEMP,HGHT_1*1000,'spline','extrap');
        TEMP_1=TEMP_1';
        HGHT_1=HGHT_1'*1000;
        a=size(HGHT_1,1);
        out_1{1,k}=date{1,i};
        for i_1=2:a+1
            out_1{i_1,k}=PRES_1(i_1-1,1);
            out_1{i_1,k+1}=HGHT_1(i_1-1,1);
            out_1{i_1,k+2}=TEMP_1(i_1-1,1);
            out_1{i_1,k+3}=RELH_1(i_1-1,1);
        end
            k=k+4;
    end
end
if ~exist(path_out,'dir')
    mkdir(path_out)
end
pathsplit_cell = strsplit(path_in,'\');
filename_1 = pathsplit_cell{end};
filename_cell = strsplit(filename_1,'(');
filename_2 = filename_cell{1};
xlswrite([path_out,'\',filename_2,'(≤Â÷µ∫Û)_long.xlsx'],out_1)