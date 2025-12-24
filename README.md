

export const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: '#7E57C2',
    accent: '#42A5F5',
    background: '#F5F7FA',
    surface: '#FFFFFF',
    text: '#2C3E50',
    disabled: '#95A5A6',
    placeholder: '#BDC3C7',
    backdrop: '#2C3E50',
    error: '#EF5350',
    success: '#66BB6A',
    warning: '#FFA726',
    info: '#29B6F6',
  },
  fonts: {
    regular: {
      fontFamily: 'Cairo-Regular',
      fontWeight: 'normal' as 'normal',
    },
    medium: {
      fontFamily: 'Cairo-SemiBold',
      fontWeight: '600' as '600',
    },
    light: {
      fontFamily: 'Cairo-Light',
      fontWeight: '300' as '300',
    },
    thin: {
      fontFamily: 'Cairo-ExtraLight',
      fontWeight: '200' as '200',
    },
  },
  roundness: 12,
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const typography = {
  h1: {
    fontSize: 32,
    lineHeight: 40,
    fontFamily: 'Cairo-Bold',
  },
  h2: {
    fontSize: 24,
    lineHeight: 32,
    fontFamily: 'Cairo-SemiBold',
  },
  h3: {
    fontSize: 20,
    lineHeight: 28,
    fontFamily: 'Cairo-SemiBold',
  },
  body: {
    fontSize: 16,
    lineHeight: 24,
    fontFamily: 'Cairo-Regular',
  },
  caption: {
    fontSize: 14,
    lineHeight: 20,
    fontFamily: 'Cairo-Regular',
  },
  small: {
    fontSize: 12,
    lineHeight: 16,
    fontFamily: 'Cairo-Light',
  },
};

export default theme;
