function spots_t = spots_remove(spots_t,remove_num)
spots_t.ID(remove_num) = [];
spots_t.h_position(remove_num) = [];
spots_t.w_position(remove_num) = [];
spots_t.size(remove_num) = [];
spots_t.intensity(remove_num) = [];
spots_t.velocity(remove_num) = [];
spots_t.direction(remove_num) = [];
spots_t.velocity_hold_frame(remove_num) = [];
end