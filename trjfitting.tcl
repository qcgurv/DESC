# Ask the user for the residue name of the reference structure
puts "Please enter the residue name of the reference structure:"
gets stdin resname

# Atom selection for the group to align
set selection_string "resname $resname"

# Create atom selections
set sel_reference [atomselect top "$selection_string" frame 0]
set all_reference [atomselect top "all" frame 0]

# Create a temporary atom selection for the trajectory
set sel_trajectory [atomselect top "$selection_string"]
set all_trajectory [atomselect top "all"]

# Loop through each frame in the trajectory
set num_frames [molinfo top get numframes]
for {set i 0} {$i < $num_frames} {incr i} {
    $sel_trajectory frame $i
    $all_trajectory frame $i

    # Align the trajectory to the reference structure
    set M [measure fit $sel_trajectory $sel_reference]
    $all_trajectory move $M
}

# Cleanup
$sel_reference delete
$all_reference delete
$sel_trajectory delete
$all_trajectory delete

puts "Trajectory aligned to the reference structure."
