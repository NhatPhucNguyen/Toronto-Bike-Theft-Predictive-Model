import {
    Box,
    Button,
    FormControl,
    FormLabel,
    Grid,
    InputAdornment,
    Modal,
    TextField,
    Typography,
} from "@mui/material";
import { DatePicker } from "@mui/x-date-pickers";
import { Dayjs } from "dayjs";
import { ChangeEvent, MouseEvent, useState } from "react";
import InputData from "./interface/InputData";
import { useNavigate } from "react-router-dom";
type PredictiveModelSelection = "lr" | "dt" | "rf";
type ModalValues = {
    result?: "Stolen" | "Recovered";
    isShow: boolean;
    model?: "Logistic Regression" | "Random Forest" | "Decision Tree";
    accScore?: number;
};
const modalStyle = {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: "fit-content",
    bgcolor: "background.paper",
    border: "2px solid #000",
    boxShadow: 24,
    p: 4,
    textAlign: "center",
};
const Home = () => {
    const navigate = useNavigate();
    const [model, setModel] = useState<PredictiveModelSelection>();
    const [modalValues, setModalValues] = useState<ModalValues>({
        isShow: false,
    });
    const [inputData, setInputData] = useState<InputData>({} as InputData);
    const handleClose = () => {
        setModalValues((prev) => ({ ...prev, isShow: false }));
    };
    const handleModelSelect = (e: MouseEvent<HTMLButtonElement>) => {
        const modelSelected = e.currentTarget.id as PredictiveModelSelection;
        setModel(modelSelected);
        if (modelSelected === "rf") {
            setModalValues((prev) => ({
                ...prev,
                model: "Random Forest",
                accScore: 76.12,
            }));
        }
        if (modelSelected === "dt") {
            setModalValues((prev) => ({
                ...prev,
                model: "Decision Tree",
                accScore: 99.32,
            }));
        }
        if (modelSelected === "lr") {
            setModalValues((prev) => ({
                ...prev,
                model: "Logistic Regression",
                accScore: 95.48,
            }));
        }
    };
    const handleBasicChange = (e: ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value;
        if (e.target.name === "BIKE_COST" || e.target.name.includes("HOUR")) {
            setInputData((prev) => ({
                ...prev,
                [e.target.name]: parseInt(value),
            }));
        } else {
            setInputData((prev) => ({
                ...prev,
                [e.target.name]: value,
            }));
        }
    };
    const handleLongLatChange = (e: ChangeEvent<HTMLInputElement>) => {
        const value = parseFloat(e.target.value);
        setInputData((prev) => ({ ...prev, [e.target.name]: value }));
    };
    const daysIntoYear = (date: Date) => {
        return (
            (Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()) -
                Date.UTC(date.getFullYear(), 0, 0)) /
            24 /
            60 /
            60 /
            1000
        );
    };
    const setDate = (value: Dayjs | null, name: "occur" | "report") => {
        const days = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ];
        if (value) {
            const day = value.daysInMonth();
            const year = value.year();
            const month = value
                .toDate()
                .toLocaleString("default", { month: "long" });
            const dow = days[value.day()];
            const doy = daysIntoYear(value.toDate());
            if (name === "occur") {
                setInputData((prev) => ({
                    ...prev,
                    OCC_DAY: day,
                    OCC_YEAR: year,
                    OCC_MONTH: month,
                    OCC_DOW: dow,
                    OCC_DOY: doy,
                }));
            } else {
                setInputData((prev) => ({
                    ...prev,
                    REPORT_DAY: day,
                    REPORT_YEAR: year,
                    REPORT_MONTH: month,
                    REPORT_DOW: dow,
                    REPORT_DOY: doy,
                }));
            }
        }
    };
    return (
        <main>
            <Box
                component={"form"}
                margin={"auto"}
                marginTop={5}
                marginBottom={2}
                width={700}
                boxShadow={5}
                padding={1}
                onSubmit={async (e) => {
                    e.preventDefault();
                    const arr = [inputData];
                    const myHeaders = new Headers();
                    myHeaders.append("Content-Type", "application/json");
                    if (model) {
                        const response = await fetch(
                            `http://localhost:3000/predict/${model}`,
                            {
                                method: "POST",
                                body: JSON.stringify(arr),
                                headers: myHeaders,
                            }
                        );
                        if (response.ok) {
                            const data = await response.json();
                            if (data.trace) {
                                alert("Please select enter required fields");
                                navigate("/");
                            } else {
                                setModalValues((prev) => ({
                                    ...prev,
                                    isShow: true,
                                    result:
                                        data.prediction[1] == 0
                                            ? "Stolen"
                                            : "Recovered",
                                }));
                            }
                        }
                    } else {
                        alert("Please select a model");
                    }
                }}
            >
                <Typography
                    textAlign={"center"}
                    marginBottom={3}
                    fontSize={20}
                    sx={{ fontWeight: "bold" }}
                >
                    Predicting Bike Status
                </Typography>
                <FormControl sx={{ display: "block", marginBottom: 1 }}>
                    <FormLabel>Select predictive model:</FormLabel>
                    <Grid
                        width={"90%"}
                        margin={"auto"}
                        justifyContent={"center"}
                        container
                        columnSpacing={1}
                        columns={3}
                        marginTop={1}
                    >
                        <Grid item xs={1}>
                            <Button
                                fullWidth
                                onClick={handleModelSelect}
                                id="lr"
                                variant={model === "lr" ? "outlined" : "text"}
                            >
                                Logistic Regression
                            </Button>
                        </Grid>
                        <Grid item xs={1}>
                            <Button
                                fullWidth
                                onClick={handleModelSelect}
                                id="rf"
                                variant={model === "rf" ? "outlined" : "text"}
                            >
                                Random Forest
                            </Button>
                        </Grid>
                        <Grid item xs={1}>
                            <Button
                                fullWidth
                                onClick={handleModelSelect}
                                id="dt"
                                variant={model === "dt" ? "outlined" : "text"}
                            >
                                Decision Tree
                            </Button>
                        </Grid>
                    </Grid>
                </FormControl>
                <Grid
                    container
                    margin={"auto"}
                    width={"100%"}
                    columns={2}
                    spacing={1}
                >
                    <Grid xs={1} item>
                        <TextField
                            type="text"
                            label="Primary Offense"
                            fullWidth
                            name="PRIMARY_OFFENCE"
                            value={inputData.PRIMARY_OFFENCE || ""}
                            onChange={handleBasicChange}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="text"
                            label="Premises Type"
                            fullWidth
                            name="PREMISES_TYPE"
                            value={inputData.PREMISES_TYPE || ""}
                            onChange={handleBasicChange}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <DatePicker
                            label="Occur Date"
                            sx={{ width: "100%" }}
                            onChange={(value) => {
                                setDate(value, "occur");
                            }}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            label="Occur Hour"
                            sx={{ width: "100%" }}
                            type="number"
                            name="OCC_HOUR"
                            value={inputData.OCC_HOUR || ""}
                            onChange={handleBasicChange}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <DatePicker
                            label="Report Date"
                            sx={{ width: "100%" }}
                            onChange={(value) => {
                                setDate(value, "report");
                            }}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            label="Report Hour"
                            sx={{ width: "100%" }}
                            name="REPORT_HOUR"
                            value={inputData.REPORT_HOUR || ""}
                            onChange={handleBasicChange}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="text"
                            label="Division"
                            fullWidth
                            name="DIVISION"
                            value={inputData.DIVISION || ""}
                            onChange={handleBasicChange}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="text"
                            label="Location Type"
                            fullWidth
                            name="LOCATION_TYPE"
                            value={inputData.LOCATION_TYPE || ""}
                            onChange={handleBasicChange}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="text"
                            label="Bike Make"
                            fullWidth
                            name="BIKE_MAKE"
                            value={inputData.BIKE_MAKE || ""}
                            onChange={handleBasicChange}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="text"
                            label="Bike Model"
                            fullWidth
                            name="BIKE_MODEL"
                            value={inputData.BIKE_MODEL || ""}
                            onChange={handleBasicChange}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="text"
                            label="Bike Type"
                            fullWidth
                            name="BIKE_TYPE"
                            value={inputData.BIKE_TYPE || ""}
                            onChange={handleBasicChange}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="text"
                            label="Bike Color"
                            fullWidth
                            name="BIKE_COLOUR"
                            value={inputData.BIKE_COLOUR || ""}
                            onChange={handleBasicChange}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="number"
                            label="Bike Cost"
                            fullWidth
                            name="BIKE_COST"
                            value={inputData.BIKE_COST || ""}
                            onChange={(e) => {
                                const value = parseInt(e.target.value);
                                setInputData((prev) => ({
                                    ...prev,
                                    BIKE_COST: value,
                                }));
                            }}
                            InputProps={{
                                startAdornment: (
                                    <InputAdornment position="start">
                                        $
                                    </InputAdornment>
                                ),
                            }}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="number"
                            label="Bike Speed"
                            fullWidth
                            name="BIKE_SPEED"
                            value={inputData.BIKE_SPEED || ""}
                            onChange={(e) => {
                                const value = parseInt(e.target.value);
                                setInputData((prev) => ({
                                    ...prev,
                                    BIKE_SPEED: value,
                                }));
                            }}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="number"
                            label="Longitude"
                            fullWidth
                            name="LONGITUDE"
                            value={inputData.LONGITUDE || ""}
                            onChange={handleLongLatChange}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="number"
                            label="Latitude"
                            fullWidth
                            name="LATITUDE"
                            value={inputData.LATITUDE || ""}
                            onChange={handleLongLatChange}
                        />
                    </Grid>
                </Grid>
                <Grid container columns={1} marginTop={2}>
                    {/* <Grid item xs={1}>
                        <Button
                            variant="contained"
                            sx={{
                                margin: "auto",
                                display: "block",
                                width: "80%",
                            }}
                            color="warning"
                        >
                            Auto Fill
                        </Button>
                    </Grid> */}
                    <Grid item xs={1}>
                        <Button
                            variant="contained"
                            type="submit"
                            sx={{
                                margin: "auto",
                                display: "block",
                                width: "80%",
                            }}
                        >
                            Submit
                        </Button>
                    </Grid>
                </Grid>
            </Box>

            <Modal
                open={modalValues.isShow}
                onClose={handleClose}
                aria-labelledby="modal-modal-title"
                aria-describedby="modal-modal-description"
            >
                <Box sx={modalStyle}>
                    <Typography
                        id="modal-modal-title"
                        variant="h5"
                        component="h2"
                    >
                        Predicted Result By{" "}
                        {
                            <Typography
                                fontWeight={"bold"}
                                fontSize={"inherit"}
                                component={"span"}
                                color={"blue"}
                            >
                                {modalValues.model}
                            </Typography>
                        }
                    </Typography>
                    <Typography
                        id="modal-modal-title"
                        variant="h6"
                        component="h3"
                    >
                        Accuracy Score: {modalValues.accScore}%
                    </Typography>
                    <Typography id="modal-modal-description" sx={{ mt: 2,fontWeight:"bold",fontSize:23 }}>
                        The status of your bike is{" "}
                        {
                            <Typography
                                component={"span"}
                                fontWeight={"bold"}
                                fontSize={"inherit"}
                                color={
                                    modalValues.result === "Recovered"
                                        ? "green"
                                        : "red"
                                }
                            >
                                {modalValues.result}
                            </Typography>
                        }
                    </Typography>
                    <Box sx={{ mt: 2, display: "flex",gap:3 }}>
                        <Box component={"figure"} boxShadow={2} padding={1}>
                            <img src={`./${model}_cf.png`} width={300} height={260}/>
                            <Typography component={"figcaption"}>Confusion Matrix</Typography>
                        </Box>
                        <Box component={"figure"} boxShadow={2} padding={1}>
                            <img src={`./${model}_roc.png`} width={300} height={260} />
                            <Typography component={"figcaption"}>ROC Curve</Typography>
                        </Box>
                        {model !== "lr" && <Box component={"figure"} boxShadow={2} padding={1}>
                            <img src={`./${model}_imp.png`} width={300} height={260} />
                            <Typography component={"figcaption"}>Important Features</Typography>
                        </Box>}
                    </Box>
                </Box>
            </Modal>
        </main>
    );
};

export default Home;
