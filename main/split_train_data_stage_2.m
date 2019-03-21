function split_train_data_stage_2
temp = [];
index = 1;
for i = 1:37
    i
    tic;
    path = ['train_data_stage_2_all/train_stage_2_all_data_',num2str(i),'.mat'];
    load(path);
    temp = [temp;all_Training_data];
    while(size(temp,1)>400)
       Training_data = temp(1:400,:);
       savepath = ['train_data_stage_2/train_stage_2_data_',num2str(index),'.mat'];
       index = index+1;
       save(savepath,'Training_data');
       temp(1:400,:) = [];
    end
    toc;
end
Training_data = temp;
savepath = ['train_data_stage_2/train_stage_2_data_',num2str(index),'.mat'];
save(savepath,'Training_data');