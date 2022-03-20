function spots_t = spots_para_update(spots_t,spots_ini)
%% 尺寸
spots_t.size = spots_t.size + rand(size(spots_t.size)) - 0.5;
logical_temp = spots_t.size < spots_ini.size_range(1);
spots_t.size(logical_temp) = spots_t.size(logical_temp) + 1;
logical_temp = spots_t.size > spots_ini.size_range(2);
spots_t.size(logical_temp) = spots_t.size(logical_temp) - 1;

%% 中心亮度
spots_t.intensity = spots_t.intensity + 10*rand(size(spots_t.intensity)) - 0.5;
logical_temp = spots_t.intensity < spots_ini.intensity_range(1);
spots_t.intensity(logical_temp) = spots_t.intensity(logical_temp) + 10;
logical_temp = spots_t.intensity > spots_ini.intensity_range(2);
spots_t.intensity(logical_temp) = spots_t.intensity(logical_temp) - 10;

%% 方向

% 速度越大，方向改变的程度越低
spots_t.direction = spots_t.direction + ...
    (randi([0 1],size(spots_t.direction))*2 - 1)...
    .*abs(1 - (spots_t.velocity/spots_ini.velocity_range(2)).^0.5)...
    .*rand(size(spots_t.direction))*2*pi;
spots_t.direction = mod(spots_t.direction,2*pi);

%% 速度

% velocity_hold_frame 标记为 -1 ，意味着从超过了高阈值若干帧，此时必定减速
logical_temp = spots_t.velocity_hold_frame == -1;
spots_t.velocity(logical_temp) = spots_t.velocity(logical_temp) ...
	- spots_ini.velocity_h_threshold/3*rand(size(spots_t.velocity(logical_temp)));

% velocity_hold_frame 标记为 -2 ，意味着低于低阈值，此时可随机加减速(默认状态)
logical_temp = spots_t.velocity_hold_frame == -2;
spots_t.velocity(logical_temp) = spots_t.velocity(logical_temp) ...
	+ rand(size(spots_t.velocity(logical_temp))) - 0.5;

% velocity_hold_frame 标记为 0（或大于0） ，意味着从低速越过低阈值，此时必定加速
logical_temp = spots_t.velocity_hold_frame >= 0;
spots_t.velocity(logical_temp) = spots_t.velocity(logical_temp) ...
	+ spots_ini.velocity_h_threshold/3*rand(size(spots_t.velocity(logical_temp)));

% 保证速度在要求范围内
logical_temp = spots_t.velocity < spots_ini.velocity_range(1);
spots_t.velocity(logical_temp) = spots_t.velocity(logical_temp) + 1;
logical_temp = spots_t.velocity > spots_ini.velocity_range(2);
spots_t.velocity(logical_temp) = spots_t.velocity(logical_temp) - 1;

end