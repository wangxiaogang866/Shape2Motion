function generate_stage_2_train_data
FindFiles = 'train_data/';
Files = dir(fullfile(FindFiles,'*.mat'));
filenames = {Files.name}';
all_Training_data = {};
for k = 1:37
    all_Training_data = {};
    k
    temp_name = filenames{k};
    load_path_1 = [FindFiles temp_name];
    load(load_path_1);
    temp_name = temp_name(10:length(temp_name)-4);
    load_path2 = ['./train_stage_1_result/test_pred_' temp_name '.mat'];
    load(load_path2);
    a = size(Training_data,1);
    pred_simmat_val = mat2cell(pred_simmat_val,ones(a,1));
    pred_conf_logits_val = mat2cell(pred_conf_logits_val,ones(a,1));
    pred_labels_direction_val = mat2cell(pred_labels_direction_val,ones(a,1));
    pred_labels_key_p_val = mat2cell(pred_labels_key_p_val,ones(a,1));
    pred_labels_type_val = mat2cell(pred_labels_type_val,ones(a,1));
    pred_regression_direction_val = mat2cell(pred_regression_direction_val,ones(a,1));
    pred_regression_position_val = mat2cell(pred_regression_position_val,ones(a,1));
    tt_Training_data = cellfun(@(x,y,z,u,v,w,m) solve(x,y,z,u,v,w,m),Training_data,pred_simmat_val,pred_labels_key_p_val,pred_labels_direction_val,pred_regression_direction_val,pred_regression_position_val,pred_labels_type_val,'Unif',0);
    for i = 1:size(tt_Training_data,1)
        temp = tt_Training_data{i};
        dof_pred = temp.dof_pred;
        dof_mask = temp.dof_mask;
        inputs_all = temp.inputs_all;
        proposal_nx = temp.proposal_nx;
        dof_socre = temp.dof_score;
        GT_proposal_nx = temp.GT_proposal_nx;
        GT_dof = temp.GT_dof;
        len = size(proposal_nx,1);
        for j = 1:len
            temp_training_data.dof_pred = dof_pred;
            temp_training_data.dof_mask = dof_mask;
            temp_training_data.GT_dof = GT_dof(j,:);
            temp_training_data.inputs_all = inputs_all;
            temp_training_data.proposal_nx = proposal_nx(j,:);
            temp_training_data.dof_score = dof_socre(j,:);
            temp_training_data.GT_proposal_nx = GT_proposal_nx(j,:);
            all_Training_data = [all_Training_data;temp_training_data];
        end    
    end
    save_path = ['train_data_stage_2_all/train_stage_2_all_',temp_name,'.mat'];
    save(save_path,'all_Training_data','-v7.3');
end
end
function Training_data = solve(model_pred,pred_simmat,pred_labels_key_p_val,pred_labels_direction_val,pred_regression_direction_val,pred_regression_position_val,pred_labels_type_val)
tic
pred_labels_type_val = reshape(pred_labels_type_val,4096,4);
pred_labels_type_val = pred_labels_type_val(:,2:3);
[~,motion_type_label] = max(pred_labels_type_val,[],2);
pred_labels_key_p_val = reshape(pred_labels_key_p_val,4096,2);
pred_labels_direction_val = reshape(pred_labels_direction_val,4096,15);
pred_labels_direction_val = pred_labels_direction_val(:,2:15);
pred_regression_direction_val = reshape(pred_regression_direction_val,4096,3);
pred_regression_position_val = reshape(pred_regression_position_val,4096,3);
iii = exp(pred_labels_key_p_val);
sum_i = iii(:,1)+iii(:,2);
sum_all = [sum_i,sum_i];
softmax = iii./sum_all;
all_direction = model_pred.all_direction_kmeans;
input = model_pred.inputs_all(:,1:3);
core_idx = find(softmax(:,2)>0.5);
[~,pred_labels_direction] = max(pred_labels_direction_val,[],2);
core_pred_labels_direction= pred_labels_direction(core_idx);
core_pred_regression_direction = pred_regression_direction_val(core_idx,:);
core_pred_regression_position = pred_regression_position_val(core_idx,:);
position = input(core_idx,:);
direction = all_direction(core_pred_labels_direction,:);
axis_p = position + core_pred_regression_position;
axis_v = direction + core_pred_regression_direction;
pred_simmat = reshape(pred_simmat,4096,4096);
s_mat = zeros(4096);
s_mat(pred_simmat<=70)=1;
s_mat = unique(s_mat,'rows');
GT_proposal = model_pred.proposal;
GT_proposal(1,:) = [];
trun_idx = find(sum(GT_proposal,2)==0);
GT_proposal = mat2cell(GT_proposal,ones(size(GT_proposal,1),1));
[score] = cellfun(@(x) compute_score(x,s_mat),GT_proposal,'Unif',0);
score = score';
score = cell2mat(score);
[max_score,score_id] = max(score,[],2);
score_id(max_score<0.3) = [];
s_mat(max_score<0.3,:)=[];
max_score(max_score<0.3) = [];
unique_id = unique(score_id);
proposal_mask_index = [];
for i = 1:size(unique_id,1)
   temp_id = unique_id(i);
   temp_index = find(score_id ==temp_id);
   temp_max_score = max_score(temp_index);
   temp_score_id = score_id(temp_index);
   [aaa,bbb] = sort(temp_max_score,'descend');
   min_num = min(15,size(aaa,1));
   for j = 1:min_num
       if(aaa(j)>=0.4)
           proposal_mask_index = [proposal_mask_index;temp_index(bbb(j))];
       end
   end
end
score_id = score_id(proposal_mask_index);
size(find(score_id~=0))
proposal_nx = s_mat(proposal_mask_index,:);
GT_dof = model_pred.dof_matrix;
GT_dof_cell = mat2cell(GT_dof,ones(size(GT_dof,1),1));
GT_Points_all_pred_dof_mat=cellfun(@(x) Rotation3D(x,input),GT_dof_cell,'Unif',0); 
Pred_dof_all = [axis_p,axis_v,motion_type_label(core_idx)];
Pred_dof_all_cell = mat2cell(Pred_dof_all,ones(size(Pred_dof_all,1),1));
Points_all_pred_dof_mat=cellfun(@(x) Rotation3D(x,input),Pred_dof_all_cell,'Unif',0);
score_id_cell = mat2cell(score_id,ones(size(score_id,1),1));
dof_score = cellfun(@(x) compute_dof_score(x,cell2mat(GT_proposal),GT_Points_all_pred_dof_mat,Points_all_pred_dof_mat,trun_idx,core_idx),score_id_cell,'Unif',0);
dof_mask = zeros(4096,1);
dof_mask(core_idx) = 1;
Training_data.dof_mask = dof_mask; 
Training_data.proposal_nx = logical(proposal_nx); 
Training_data.dof_score = cell2mat(dof_score);
Training_data.inputs_all = model_pred.inputs_all;
dof_pred = zeros(4096,7);
dof_pred(core_idx,:) = Pred_dof_all;
Training_data.dof_pred = dof_pred;
temp_proposal = model_pred.proposal;
temp_proposal(1,:) = [];
Training_data.GT_proposal_nx = temp_proposal(score_id,:);
Training_data.GT_dof = GT_dof(score_id,:);
%vis(Training_data);

toc;
end

function dof_score = compute_dof_score(score_id_cell,GT_proposal,GT_Points_all_pred_dof_mat,Points_all_pred_dof_mat,turn_idx,core_idx)
    if score_id_cell == 0
        dof_score = zeros(4096,1);
        dof_score = dof_score';
    else
        idx = find(GT_proposal(score_id_cell,:)==1);
        GT_mat = GT_Points_all_pred_dof_mat{score_id_cell};
        temp_GT  = GT_mat(idx',:);
        dof_score_step = cellfun(@(x) compute_dof_score_step(idx,temp_GT,x),Points_all_pred_dof_mat,'Unif',0);
        dof_score_step = cell2mat(dof_score_step);
        dof_score = zeros(4096,1);
        dof_score(core_idx) = dof_score_step;
        dof_score =dof_score';
        have_turn = find(turn_idx == score_id_cell+1);
        if have_turn
            GT_mat_2 = GT_Points_all_pred_dof_mat{score_id_cell+1};
            temp_GT_2  = GT_mat_2(idx',:);
            dof_score_step_2 = cellfun(@(x) compute_dof_score_step(idx,temp_GT_2,x),Points_all_pred_dof_mat,'Unif',0);
            dof_score_step_2 = cell2mat(dof_score_step_2);
            dof_score_2 = zeros(4096,1);
            dof_score_2(core_idx) = dof_score_step_2;
            dof_score_2 =dof_score_2';
            dof_score = min(dof_score,dof_score_2);
        end
    end
end

function dof_score = compute_dof_score_step(idx,GT_mat,pred_mat)
    temp_pred  = pred_mat(idx',:);
    temp = GT_mat-temp_pred;
    dof_score = mean(sum(abs(temp).^2,2).^(1/2));
    dof_score = -2/(1+exp(-4*dof_score))+2;
end

function [score] = compute_score(GT_proposal,s_mat)
GT_proposal = repmat(GT_proposal,size(s_mat,1),1);
union = logical(GT_proposal+s_mat);
inter = GT_proposal.*s_mat;
score = sum(inter,2)./sum(union,2);
end


function points_rot=Rotation3D(Dof_para,Points)
   if Dof_para(7)==1 
     theta=pi;
     points_rot=rot3d(Points,Dof_para(1:3),Dof_para(4:6),theta); 
   elseif Dof_para(7)==2
      scale=1;  
      points_rot=trans3d(Points,Dof_para(4:6),scale);
   elseif Dof_para(7)==3
      scale=1;
      theta=pi;
      points_rot=trans3d(Points,Dof_para(4:6),scale);
      points_rot=rot3d(points_rot,Dof_para(1:3),Dof_para(4:6),theta);
   end
end

function Pr=rot3d(P,origin,dirct,theta)
    dirct=dirct(:)/norm(dirct);
    A_hat=dirct*dirct';
    A_star=[0,         -dirct(3),      dirct(2)
            dirct(3),          0,     -dirct(1)
           -dirct(2),   dirct(1),            0];
    I=eye(3);
    M=A_hat+cos(theta)*(I-A_hat)+sin(theta)*A_star;
    origin=repmat(origin(:)',size(P,1),1);
    Pr=(P-origin)*M'+origin;
     
end

function point_trans=trans3d(point,direct,scale)
   direct=direct(:)/norm(direct);
   direct1=scale*direct;
   point_trans=point+repmat(direct1',size(point,1),1);
end
