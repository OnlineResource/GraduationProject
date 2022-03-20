function [spots,spots_ini] = spots_evo(spots,size_frame,spots_ini,t)
spots(t) = spots(t - 1);

% 随机消失
if rand() < spots_ini.die_rate && ~isempty(spots(t).ID)
    die_num = randi(length(spots(t).ID));
    spots(t) = spots_remove(spots(t),die_num);
end

% 位置等参数更新
spots(t) = spots_move(spots(t),spots_ini);

% 从图像边缘移出
spots(t) = spots_move_out(spots(t),size_frame);

% 尺寸、中心亮度、方向、速度变化
spots(t) = spots_para_update(spots(t),spots_ini);

% 随机从图像边缘移入
if rand() < spots_ini.move_in_rate
    spot_one = spots_move_in(size_frame,spots_ini);
    spots_ini.ID_total = spots_ini.ID_total + 1;
    spots(t) = spots_add(spots(t),spot_one,spots_ini.ID_total);
end

spots(t).spots_num = length(spots(t).ID);

end