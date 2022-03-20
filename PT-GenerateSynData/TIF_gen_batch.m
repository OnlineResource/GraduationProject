clear;
addpath(genpath(pwd))

gen_num = 100;
save_path = 'synthesized_data_50_frames_per_file_150i200-1700_30n50_10d20';  % 记得修改 目录 和 最大亮度 ！！！
if exist(save_path,'dir')
    error(['The directory "' save_path '" exists!']);
else
	mkdir(save_path);
end
H = 500;W = 500; % bit_depth = 16;
T = 50; % 持续帧数
size_frame = [H,W,T];
background_range = [80 120];

spots_ini.quantity_range = [30 50]; % 初始数量范围
spots_ini.size_range = [10 20]; % 直径范围
spots_ini.intensity_range = [150 1700]; % 中心亮度范围
spots_ini.velocity_range = [0 10]; % 速度范围（隔帧位移）
spots_ini.velocity_h_threshold = 9; % 较大速度阈值（超过后开始计算帧数）
spots_ini.velocity_l_threshold = 2; % 较小速度阈值（超过后开始持续加速）
spots_ini.velocity_hold = 2; % 较大速度保持帧数
spots_ini.move_in_rate = 0.4; % 从帧范围外 移入帧内的概率
spots_ini.die_rate = 0; % 从帧范围内 变为不可视的概率
spots_ini.ID_total = 0; % 初始化 ID 数

for idx=1:gen_num
    spots_ini.intensity_range = [150 10*randi([20, 170])]; % 中心亮度范围
    disp([num2str(idx) '/' num2str(gen_num)])
    % 生成 T 帧的背景
    background = background_gen(size_frame,background_range);
    
    % 生成 T 帧的目标数据
    spots = spots_gen(size_frame,spots_ini);
    
    % 合成 T 帧的图像
    frame = frame_synthesize(background,spots,spots_ini);
    
    for s_idx=1:length(spots)
        % 修正 MATLAB 索引起始为 1 带来的偏差
        spots(s_idx).h_position = spots(s_idx).h_position - 1;
        spots(s_idx).w_position = spots(s_idx).w_position - 1;
    end
    
    file_name = ['syn_' datestr(now,'yyyy_mm_dd__HH_MM_SS_FFF')];
    imwriteGrayTiff(frame, [save_path '/' file_name  '.tif'])
    save([save_path '/' file_name],'spots','spots_ini')
    csv_save(spots, [save_path '/' file_name '.csv'])
end