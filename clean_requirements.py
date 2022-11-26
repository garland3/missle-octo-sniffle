import os
# import multiprocessing

ignore_items = ['anaconda', 'missle', 'conda','menuinst', 'mkl', 'navigator', 'curl', 'pywin']

if __name__ == '__main__':
        
    def parse_requirement(req):
        requirement = req.strip()
        print(requirement)
      
        for item in ignore_items:
            if requirement.find(item) != -1:
                print(f"Removing {requirement}")
                return None
      
        idx = requirement.find("==")
        if idx ==-1:
            print(f"No version found for {requirement}")
            
        requirement = requirement[:idx]
        return requirement
        
    with open("requirements_base.txt", "r") as f:
        requirements = f.readlines()
        
        
    new_requirements = []
    for requirement_raw in requirements:
        requirement = parse_requirement(requirement_raw)
        if requirement is not None:
            new_requirements.append(requirement)
        
    # print(new_requirements)
        
    with open("requirements.txt", "w") as f:
        # f.write
        
        # f.writelines(new_requirements)
        for r in new_requirements:
            f.write(f"{r}\n")
            
        