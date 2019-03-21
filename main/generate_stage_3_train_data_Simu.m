function generate_stage_3_data_Simu
FindFiles = 'train_data_stage_2/';
Files = dir(fullfile(FindFiles,'*.mat'));
filenames = {Files.name}';
for i = 111:size(filenames,1)
tic;
temp_name = filenames{i};
load_path = [FindFiles temp_name];
load(load_path)
Training_data = cellfun(@(x) solve(x),Training_data,'Unif',0);
save_path = ['train_data_stage_3/train_data_stage_3_' temp_name(15:end-4) '.mat'];
save(save_path,'Training_data');
toc;
end
end

function data = solve(Training_data)
dof_pred = Training_data.dof_pred;
dof_score = Training_data.dof_score;
inputs_all = Training_data.inputs_all;
GT_dof = Training_data.GT_dof;
GT_proposal_nx = Training_data.GT_proposal_nx;
proposal_nx = Training_data.proposal_nx;
[pred_max_score,pred_dof_index] = sort(dof_score,'descend');
pred_dof_index = pred_dof_index(1:10);
c=randperm(numel(pred_dof_index));
dof_index=pred_dof_index(c(1));
pred_dof_net = dof_pred(dof_index,:);
data.inputs_all = inputs_all;
data.proposal = GT_proposal_nx;
data.dof_regression = GT_dof(1:6)-pred_dof_net(1:6);
temp_proposal = proposal_nx;
temp_point = inputs_all(temp_proposal==1,1:3);
field = zeros(3,4096,6);
for j = 1:3
    temp_field = inputs_all;
    angle = 30*j;
    rot_point = Rotation3D(pred_dof_net,temp_point,angle);
    temp_field(temp_proposal==1,1:3) = rot_point;
    field(j,:,:) =temp_field ;
end
data.field = field;
data.s2_proposal = proposal_nx;
%vis_data(data);
end

function vis_data(data)
inputs = data.inputs_all;
inputs = inputs(:,1:3);
field = data.field;
field1 = field(1,:,1:3);
field2 = field(2,:,1:3);
field3 = field(3,:,1:3);
field1 = reshape(field1,4096,3);
field2 = reshape(field2,4096,3);
field3 = reshape(field3,4096,3);
proposal = inputs(data.proposal==1,:);
figure(1)
plot3(proposal(:,1),proposal(:,2),proposal(:,3),'o');
axis equal;
close(figure(1));
figure(1)
plot3(inputs(:,1),inputs(:,2),inputs(:,3),'o');
axis equal;
close(figure(1));
figure(1)
plot3(field1(:,1),field1(:,2),field1(:,3),'o');
axis equal;
close(figure(1));
figure(1)
plot3(field2(:,1),field2(:,2),field2(:,3),'o');
axis equal;
close(figure(1));
figure(1)
plot3(field3(:,1),field3(:,2),field3(:,3),'o');
axis equal;
close(figure(1));
end

function points_rot=Rotation3D(Dof_para,Points,angle)
   if Dof_para(7)==1 
     theta=angle*pi/180;
     points_rot=rot3d(Points,Dof_para(1:3),Dof_para(4:6),theta); 
   elseif Dof_para(7)==2 
      scale=0.2;  
      points_rot=trans3d(Points,Dof_para(4:6),scale);
   elseif Dof_para(7)==3 
      scale=0.2;
      theta=angle*pi/180;
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
