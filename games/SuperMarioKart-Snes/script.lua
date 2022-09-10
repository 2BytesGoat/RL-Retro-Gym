function getLap()
	-- 133 final
	return data.kart1_lap-128
end

function getMilisec()
	return data.curr_miliSec - 300
end

function getCheckpoint()
	local checkpoint = data.kart1_checkpoint
	local lapsize = data.lapsize
	local lap = data.kart1_lap-128
	-- local rank = data.rank/2+1

	return checkpoint + (lap)*lapsize
end

function isTurnedAround()
	return data.isTurnedAround == 0x10
end


function isDone()
	if data.game_mode ~= 0x1C then
		return true
	end

	local lap = getLap()
	if lap >= 5 then --for now
		return true
	end

	return false
end

function isDoneTrain()
	return isDone() or isHittingWall()
end

wall_hits=0
wall_steps=0
function isHittingWall()
	--FIXME change name probably since is also feel of
	-- if it hits wall 5 times in less than 
	wall_steps = wall_steps + 1 

	if data.kart1_surface == 128 or data.kart1_surface == 40 or data.kart1_surface == 32 then
		-- hit a wall, or fells off (40,32) 
		wall_hits = wall_hits + 1
	end

	if data.kart1_surface==34 then
		--goes to deep water(34)
		--allows small dives 
		wall_hits= wall_hits + 0.05
	end

	if wall_hits >= 5 then
		wall_hits = 0
		wall_steps = 0
		return true
	end 

	if wall_steps == 500 then
		--if did't got enought wall hits in 500 reset
		wall_hits = 0
		wall_steps = 0
	end

	return false
end

data.prevCheckpoint = getCheckpoint() 
data.prevFrame = data.kart1_frame
function getCheckpointReward()
	local newCheckpoint = getCheckpoint()
	
	local curFrame = data.kart1_frame
	if curFrame < data.prevFrame or curFrame > data.prevFrame + 60 then
		data.prevCheckpoint = getCheckpoint()
	end
	data.prevFrame = curFrame
	
	local reward = 0
	--- this will give rewards of 100 on each cross
	reward = reward + (newCheckpoint - data.prevCheckpoint) * 10
	data.prevCheckpoint = newCheckpoint

	-- Sanity check
	if reward < -5000 then
		return 0
	end
	
	return reward
end

function getSpeedReward()
	local reward = 0
	local speed = data.kart1_speed

	if isTurnedAround() then
		reward = -0.1 --not sure if should keep
	elseif speed > 900 then
		reward = 0.2
	elseif speed > 800 then
		reward = 0.1
	elseif speed > 600 then
		reward = 0
	else
		reward = -0.1
	end
	
	return reward
end

function getSpeedRewardLess()	
	local reward = 0
	local speed = data.kart1_speed

	if isTurnedAround() then
		reward = -0.1 --not sure if should keep
	elseif speed > 900 then
		reward = 0.1
	elseif speed > 600 then
		reward = 0
	else
		reward = -0.1
	end
	
	return reward
end

function getExperimentalReward()
	local reward = 0

	if data.kart1_surface == 128 then
		-- hit a wall
		reward = -1 
	end
	if data.kart1_surface == 40 or data.kart1_surface == 32 or data.kart1_surface==34 then
		-- feel of, or deep dived
		reward = -1 
	end

	return reward
end

function getRewardTrainSpeed()
	-- 0.2 top speed, 1 passing checkpoint
	return getSpeedReward() + getCheckpointReward() + getExperimentalReward()
end

function getRewardTrainSpeedLess()
	-- 0.2 top speed, 1 passing checkpoint, force top speed
	return getSpeedRewardLess() + getCheckpointReward() + getExperimentalReward()
end

function getRewardTrain()
	-- 1 passing checkpoint
	return getCheckpointReward() + getExperimentalReward()
end
