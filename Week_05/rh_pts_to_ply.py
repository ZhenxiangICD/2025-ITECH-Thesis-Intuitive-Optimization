import Rhino
import scriptcontext
import System.IO

def export_points_to_ply(points, file_path):
    """
    Exports a list of Point3d to a .ply file.
    
    Parameters:
        points (list): List of Rhino.Geometry.Point3d.
        file_path (str): Full path to save the .ply file.
    """
    try:
        # Open the file for writing
        with open(file_path, 'w') as ply_file:
            # Write the PLY header
            ply_file.write("ply\n")
            ply_file.write("format ascii 1.0\n")
            ply_file.write(f"element vertex {len(points)}\n")
            ply_file.write("property float x\n")
            ply_file.write("property float y\n")
            ply_file.write("property float z\n")
            ply_file.write("end_header\n")
            
            # Write point data
            for point in points:
                ply_file.write(f"{point.X} {point.Y} {point.Z}\n")
        
        print(f"PLY file successfully written to: {file_path}")
    except Exception as e:
        print(f"Error writing PLY file: {e}")

# Example usage in Grasshopper:
# export_points_to_ply(pts, "C:\\path\\to\\your\\file.ply")
