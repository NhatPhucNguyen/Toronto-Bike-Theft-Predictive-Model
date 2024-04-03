import { Box, Typography, AppBar, Button, Toolbar } from '@mui/material'
const Navbar = () => {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            COMP309-Group2
          </Typography>
          <Button color="inherit">Home</Button>
          <Button color="inherit">Data Exploration</Button>
          <Button color="inherit">GitHub</Button>
        </Toolbar>
      </AppBar>
    </Box>
  )
}

export default Navbar