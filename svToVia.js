const path = require("path");
const fs = require("fs");

const svDatasetBaseFolder = path.join(__dirname, "vehicle_dataset/Vehicles");
const datasetBaseFolder = path.join(__dirname, "dataset");

console.log(svDatasetBaseFolder);
const svAnnotationsFolder = path.join(svDatasetBaseFolder, "testfiles/ann");
const svImagesFolder = path.join(svDatasetBaseFolder, "testfiles/img");

const annotationFiles = fs.readdirSync(svAnnotationsFolder).sort();
const imageFiles = fs.readdirSync(svImagesFolder).sort();
const trainDataSize = Math.floor(annotationFiles.length * 0.9);
const valDataSize = annotationFiles.length - trainDataSize;

console.log(annotationFiles);

prepare("train", 0, trainDataSize);
prepare("val", trainDataSize - 1, valDataSize);

function prepare(stage, startIndex, size) {
    const viaRegionData = {};
    console.log("Prepare", stage, startIndex, size);
    try {fs.mkdirSync(path.join(datasetBaseFolder, stage));}
    catch(error) { console.log("Directory Exists")}
    annotationFiles.slice(startIndex, startIndex + size).forEach(annotationFile => {
        const annotation = JSON.parse(fs.readFileSync(path.join(svAnnotationsFolder, annotationFile), "utf-8"));
        console.log(annotationFile);
        
        const stats = fs.statSync(path.join(svAnnotationsFolder, annotationFile));
        const annotationForVia = {
            fileref: "",
            size: stats.size,
            filename: annotationFile.replace(".json", ""),
            base64_img_data: "",
            file_attributes: {},
            // accomodate multiple
            regions: {
                "0": {
                    shape_attributes: {
                        name: "polygon",
                        all_points_x: annotation.objects[1].points.exterior.map(point => point[0]),
                        all_points_y: annotation.objects[1].points.exterior.map(point => point[1])
                    },
                    region_attributes: {}
                }
            }
        }
        fs.copyFileSync(path.join(svImagesFolder, annotationForVia.filename), path.join(datasetBaseFolder, stage, annotationForVia.filename));
        viaRegionData[annotationForVia.filename + annotationForVia.size] = annotationForVia;
    });

    fs.writeFileSync(path.join(datasetBaseFolder, stage, "via_region_data.json"), JSON.stringify(viaRegionData, null, 4));
}

