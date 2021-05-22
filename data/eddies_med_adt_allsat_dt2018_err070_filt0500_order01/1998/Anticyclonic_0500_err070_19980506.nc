CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�M����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��#      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �8Q�   max       =�/      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @FG�z�H     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
=�   max       @vH�\)     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @M@           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @��           �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >�        �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��y   max       B0и      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�x�   max       B1@      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�X   max       C��      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�.   max       C��      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P'��      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�`�d��8   max       ?�I�^5@      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �49X   max       >%      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�   max       @FG�z�H     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @vG��Q�     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @M@           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @�Ҁ          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�IQ���   max       ?�
�L/�|     �  N�         7      0   _                                 _         
         "            H   	         <      )   (      	   	   5   K                              2      �         �O/3�N��PpNNd�P �P��*O�ǢO�Nr�(O��fOSN&a�OV�KN��IO�0�N�vP��#OL��O#UN�1/OF��O�;�P.�IO.[�NhdN��P$+N���Ne�YN]�bO�+�Od��P.ȌOI��O�B�N��lN�D�O�/'O��@N�c�N��N�O�N
�HN�5�N��NgҸN��O���M��O��Ob��N3W�O�Y�8Q�e`B��o;D��;��
<o<o<t�<T��<�o<�o<�o<�C�<���<��
<��
<�1<�1<�1<�9X<ě�<���<���<�h<��<��=o=o=\)=#�
=,1=,1=49X=D��=L��=L��=P�`=Y�=Y�=Y�=e`B=m�h=q��=y�#=}�=}�=��=�\)=��=��w=��=�1=�
==�/�~������������������pkjkqtv������tpppppp������(����������������������������/;H[_``^aT;/"�������
$!-6-
���"$1<IUbn{|�{wpbU<0'"[X[_gtz�����yutgcca[���������������������������������������������������������������������������������������������������.*.6?BMORTTOB?76....����������������������������������������������5FTWWNB����:7BHN[]gltx}wtg[NCB:��������������������������

���XVamz���������zxmgaX%)3BN\efaTNB5)fopu�������������tgf�������������������������������������������

##)**%#
	������������$,.'���������������	���������mmnz�������zummmmmmm��������������������#(/<=BXdeaUH/#.$'-<HX[STR^]_UHC<5.������)5IJ?)�������������������������4DN[t�����ztnf[N854degimst������ytgdddd����������������	)5AGLJB)��������)14)������56<>BCO[bg[YOFB65555WPX[ht������tih[WWWW2*467BEGB62222222222$")*6?OZ\eh\\OC@6-*$GGIPUbmb`UMIGGGGGGGG����������������������������������������w�����������wwwwwwww#,/<@=<2/+#������������������������������������������������kmo��������������zok#

########�������

�������h�t�~āĊďĐčā�t�h�[�P�O�D�E�O�[�^�h�a�n�zÇÓÖÓÊÇ�z�n�a�\�]�a�a�a�a�a�a�	�"�/�=�N�Q�/�"�����������������������	�����������������������������������������T�a�m�|��l�H�;�/��������������;�H�T��������)�N�X�X�N�B���������m�Z�W�m�������������������ɼ���������o�\�d�n����������������������������������}��������¿��������������¿²°ª²³¿¿¿¿¿¿�����������&�"���������ùðìóù���(�5�A�N�P�W�Z�Z�Z�R�N�A�5�5�(�#�� �(�(�f�s�������s�f�^�\�f�f�f�f�f�f�f�f�f�fŭůŹſ������������ŹŭŠŒőŒŔŠŪŭ���ʾ׾ھ׾־ʾȾ������������������������O�\�h�n�r�q�v�w��u�\�O�6�(�$�,�5�8�B�O�zÇÓàìòôìàÝÓÇÄ�z�x�u�z�z�z�z������0�/�$�����ƳƎ�|�zƍƎƆƈƚ����������������������������������ʼּܼؼּμμʼ����������������������M�Z�f�s�������s�f�Z�M�B�A�=�8�A�H�M�M��������������������ŹŶŭťũŭż�����߿"�.�;�G�T�Y�`�f�g�`�T�G�;�.�*�#����"�y���Ŀѿܿݿֿȿ����������`�S�P�X�c�f�y�'�.�1�3�*�'�������ݻ��������'²¿������������¿²¦¥¦±²²²²²²�������!���������ڼ��������#�<�H�T�V�T�J�<���
����������������#���������������������������������������������������������������������������������A�B�M�V�U�M�A�4�(�(�(�4�7�@�A�A�A�A�A�A��(�A�N�`�g�u�}�s�g�A�(����	������������
�����������ùìàÞáìò�����Z�g�o�q�l�^�\�U�P�N�A�(������5�N�Z�(�4�A�M�Z�f�o�q�o�p�h�f�Z�M�A�1�)�(�"�(�	�"�-�2�,�"��	�����������������������	�g�s�{�����������������s�o�g�c�b�g�g�g�g�O�P�[�h�r�t�u�u�v�t�h�[�Q�Q�O�M�O�O�O�O���
��0�<�F�Q�Q�G�<�0�#�
���������������e�~�����������������~�v�g�Y�@�9�6�>�Y�e�������������������������������������������������������������������������������������ʼʼʼ¼����������������������������m�y�����������y�x�m�`�T�T�Q�S�T�Y�`�e�m�������� ������������������!�-�.�-�'�!���������������m�y�����y�m�`�_�`�c�m�m�m�m�m�m�m�m�m�m�<�E�H�M�H�H�<�/�*�*�/�2�<�<�<�<�<�<�<�<��*�6�9�6�5�1�*�������������������ûлٻٻ׻лû��������l�e�x�z��������������������������������������������������ʼּ������ּʼ���������������ĚĝĦ������������������ĳĦĚĒċĊčĚ�n�b�U�P�P�U�b�l�n�{�{�{�o�n�n�n�n�n�n�nD�D�D�D�D�D�D�D�D�D�D�D�D�D|DvDxD�D�D�D� : H + g P h @ > ' H 0 @ 9 R C / 6 7 H b M E - C a $ 3 B Q H 6 h q 6 < Z G 2 J j 6 Z Y u ; 4 U ; A \ B f X T  v  �  �  j  {  	  �  C  y  �  7  9  �  �  �  �  �  �  }  /  �       y  �    �  #  �  �  �  	  �  �    �  �  �  {  �      ]  j  �    �  �  6  %  �  <  a  ���`B$�  =P�`<t�=Y�=��=\)<�`B<�C�=0 �=�P<���=#�
<�/=49X=��=�h=,1=o<��=49X=aG�=m�h=Y�=,1=0 �=��=#�
='�=0 �=��=�o=�{=�9X=�hs=m�h=u=��>o=u=��-=��=�hs=��=���=�o=��-=��-=��m=��
>]/=���=�S�>�  B+�B	��BB�B"
jA��yBM"B'9B	kB�-Bd�B��B �B0�B3�B�B"|BƘB��B"<�B��A�@�B��Bj�B!�BB$�B��Bp�A�ˠBu�B��B�zB�wB��B	VB	�wBv�BzB�B��BrB��B0иB'W�B��B!�B6�B"�B3B,?BK8B �#B-~B�nB)B	�B?�B" �A�x�B?�B&�#B	��B��BD�B�tB�[BA�BH�B��B"?�B�B��B"A(B��A��HBH�B
�lB!��BB$�tB�&BW�A���B��B�B��BB B�B	��B	��B��B@�B��B��B;�B�7B1@B'=#B��B ��B~�B�B?�B,@�B@B OB0�B��A���A�DA��I@E~A�@A��i@��A��A��hA��A��AB�9A��AN��BH�A�k#B��A���@��A@A�ITAd�Aq��@�ޯA��pA �A�u�A���A�8�A:��A�{�A� A���A=Z�A��*A�rIA�<)A�i?�X@?�@�;i@��iAkbH@JX@_-\Ak��A�M7A�I�@�@�A�@���A�<A�+C��A��GA�	A���@��A���A���@��A���A��)AϦ�A���AB�A���AMVB?�A�u�B�xAӁ�@��iA@�A��Ac��At�:@��A�dAȉA���A�dA���A9�A��PAѢUA���A=�A�,�A�mA�}A��?�.@#� @��@���Aju�@M�@cqAk?�A�|�A�|�@�AQ@��_A���A���C��         8      1   `                        	         `         
         "            H   	         <      *   )      	   
   6   L                              3      �         �         5      '   ;   !                              7                  +            +            !      -      %            '                                             !                                                                     '                              -      %                                                         O �BN��O/%Nd�O�FZOB��Ou�O�N7�O~�N�eN&a�OI��N��IO�>�N�vO���OL��O#UN�1/OF��O2�PN�O.[�NhdN��O��HN���Ne�YN]�bO���N��?P'��O0�mO�B�N��lNP\�O�ҲO�&tNnh�N��N�O�N
�HN�5�N��NOA�N��O���M��OD<xOb��N3W�O�-  �  �  �  )  �    �  z    b  v    y  ~  Z  �  ]  �  �    �  R  #  �  )  �  X    L  �  �  3  �  �      �  �  �  b  �  �  �  �  -  �  k  ,  	�  �  y    W  �49X�e`B<�`B;D��<�C�=�7L<e`B<t�<e`B<�1<�t�<�o<�t�<���<�1<��
=�C�<�1<�1<�9X<ě�=o<�h<�h<��<��=u=o=\)=#�
=Y�=P�`=8Q�=P�`=L��=L��=T��=u=��-=]/=e`B=m�h=q��=y�#=}�=}�=�+=�\)=��=��w=��#=�1=�
=>%������������������pkjkqtv������tpppppp����������������������������������������"/;DHTZ[[YTH;/-&$#$"��������

���,),04<Ibnstoi^UI<30,[X[_gtz�����yutgcca[����������������������������������������������������������������������������������������������������.*.6?BMORTTOB?76....��������������������������������������������)06985/����:7BHN[]gltx}wtg[NCB:��������������������������

���XVamz���������zxmgaX(")5BNQY[][ZNMB5)(kqqry�������������tk�������������������������������������������

##)**%#
	�������������������������	���������mmnz�������zummmmmmm��������������������#/<DHTZ]]UH</#*./8<?HPQHE<3/******������)BG=)��������������������������4DN[t�����ztnf[N854degimst������ytgdddd���������������)5<CIIGB;)�������� 	������=>>BFO[^e[WODB======WPX[ht������tih[WWWW2*467BEGB62222222222$")*6?OZ\eh\\OC@6-*$GGIPUbmb`UMIGGGGGGGG����������������������������������������y�����������yyyyyyyy#,/<@=<2/+#����������������������������������������� ���������kmo��������������zok#

########���������
������t�{āĉčďďčā�t�h�[�W�O�F�F�O�[�h�t�a�n�zÇÓÖÓÊÇ�z�n�a�\�]�a�a�a�a�a�a��������� �����������������������������交���������������������������������������a�n�m�a�]�I�;�/�"��� � ��	��"�/�H�a�����������������������������������������������������������������r�f�e�j�r�w��������������������������������}��������²¿��������¿²²¬²²²²²²²²²²����������������������ùöðù�����(�5�A�N�N�V�X�O�N�A�:�5�(�%��"�(�(�(�(�f�s�������s�f�^�\�f�f�f�f�f�f�f�f�f�fŔŠŭŹž������������ŹŭŨŠŝœŒœŔ���ʾ׾ھ׾־ʾȾ������������������������O�\�h�m�q�r�p�t�v�\�O�C�6�)�&�-�6�9�C�O�zÇÓàìòôìàÝÓÇÄ�z�x�u�z�z�z�z�������������������ƺƳƬƫƭƳ��������������������������������������ʼּܼؼּμμʼ����������������������M�Z�f�s�������s�f�Z�M�B�A�=�8�A�H�M�M��������������������ŹŶŭťũŭż�����߿"�.�;�G�T�]�a�b�`�T�M�G�=�;�/�.�(�#�"�"�y�����ĿѿؿٿҿĿ����������h�]�[�]�m�y�'�.�1�3�*�'�������ݻ��������'²¿������������¿²¦¥¦±²²²²²²�������!���������ڼ������������
��/�4�-�%���
���������������������������������������������������������������������������������������������������A�B�M�V�U�M�A�4�(�(�(�4�7�@�A�A�A�A�A�A��(�5�A�N�W�g�f�Z�V�A�5�(���������������������������������������������Z�g�n�p�l�^�[�T�M�A�(������,�5�N�Z�4�A�M�Z�f�m�p�m�m�f�Z�M�C�A�4�3�+�(�4�4�	�"�-�2�,�"��	�����������������������	�g�s�{�����������������s�o�g�c�b�g�g�g�g�[�h�t�u�u�t�h�[�S�S�[�[�[�[�[�[�[�[�[�[���
��#�0�A�K�J�C�<�0�#��
������������e�r�~���������������~�r�e�Y�L�G�F�L�Y�e�������������������������������������������������������������������������������������ʼʼʼ¼����������������������������m�y�����������y�x�m�`�T�T�Q�S�T�Y�`�e�m�������� ������������������!�-�.�-�'�!���������������m�y�����y�m�`�_�`�c�m�m�m�m�m�m�m�m�m�m�<�C�H�L�H�G�<�/�+�+�/�4�<�<�<�<�<�<�<�<��*�6�9�6�5�1�*�������������������ûлٻٻ׻лû��������l�e�x�z������������������������������������������������ʼּݼ����ּʼ�����������������ĚĝĦ������������������ĳĦĚĒċĊčĚ�n�b�U�P�P�U�b�l�n�{�{�{�o�n�n�n�n�n�n�nD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DzD|D�D�D� 9 H  g M + 3 > 2 H . @ 8 R F /  7 H b M 8 / C a $ ) B Q H ' + r 0 < Z % &  X 6 Z Y u ; 4 O ; A \ 5 f X R  [  �  o  j  ^  �  �  C  L      9  �  �  �  �  �  �  }  /  �  �  �  y  �      #  �  �    �  �  x    �  a  9  �  �      ]  j  �      �  6  %  �  <  a  �  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  |  �  �  w  c  K  0    �  �  �  �  t  T  6      �  �  .  �  �  �  �  |  m  ]  K  7      �  �  �  �  z  \  2  �  �  �  �  �  �  �  R  �  %  X  g  q  �  }  Z    �  /  �  �   �  )  #          �  �  �  �  �  �  �  }  _  @  #    �  �  8  h  �  �  �  �  �  �  �  �    n  C  	  �  r    �  �  �  �     q    j  �  �  �  �  �  �      �  �  ,  �  �  �  v  �  �  �  �  �  �  �  �  �  �  �  �  `  :    �  �  b  �  -  z  e  N  0    �      �  �  �  g  8    �  �  �  M  �  �  �             �  �  �  �  �  w  L  "  �  �  �  L    �  ?  O  [  `  _  T  =  !  �  �  �  �  4  �  �  k  F  0  )    q  u  v  u  p  b  Q  ?  *    �  �  �  j    �  )  �  #  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  I  x  t  l  c  T  A  (    �  �  �  �  f  8  �  �    f  4  ~  l  Z  H  5  #    �  �  �  �  �  }  N    �  �  x  i  Z  Q  Y  Q  >  "  	  �  �  �  �  q  \  T  f  \  8  �  �    �  �  �  �  �  �  �  �  j  H    �  �  l  &  �  �  T       ^  �    @  n  �  �  �    E  Z  X  ,  �  i  �  c  �  �  �  �  �  �  �  �  �  �  z  f  S  >  '    �  �  �  R    �  2  �  �  �  �  �  �  �  �  �  �  {  `  F  .  &  !  $  
  �  �  >                 �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  ]  K  1    �  �  `    �  T  �  [  �    "  0  =  N  L  :  $    �  �  �  m  )  �  c  �  ]  �  _  
    #      �  �  �  �  L    �  �  �  v  C    �  A  q  �  �  �  �  �  �  �  �  u  L    �  �  �  m    �  G  �  N  )          �  �  �  �  �  �  �  �  �  �  }  j  V  D  3  �  �  �  �  �  �  �  �  �  �  �  }  f  J  '  �  �  �  9   �  �  L  �  �  �  	  A  X  T  U  I    �  x    r  �  �  k  �        	     �  �  �  �  �  �  �  �  �  �  y  f  P  4    L  G  C  >  +    �  �  �  �  �  `  <    �  �  �  l  @    �  �  �  �  �  u  j  ^  R  F  :  .  "     �   �   �   �   �   �  �  �  �  �  �  �  �  �    Z  *  �  �  9  �  &  x  �    �  5  R  R  H  o  �  �  1  3  .  !    �  �  \    �    h  �  �  �  �  �  �  Z  )    �  �  �  J  �  �  Q    �  G  �  "  �  �  �  �  �  �  �  �  f  =    �  �  J  �  ~  �     �  &      �  �  �  l  ?    �  �  �  f  4  $    �  �  �  �  q      �  �  �  �  �  �  �  �  �  y  P  %  �  �  �  t  E    W  {  �  �  �  t  ^  D  '  
  �  �  �  �  �  �  �  u  U  4  �  �  �  �  �  �  �  �  �  X    �  Z  �  j  �  J  �    �  �  �  d  �  �  �  �  �  �  �  \  4  
  �  f  �  &  F    �  H  Q  [  `  \  X  N  ?  1      �  �  �  �  �  �  n  X  A  �  �  �  �  �  �  �  c  <    �  �  d    �  [    �  �  �  �  �  �  �  �  �  y  ^  C    �  �  j  :    �  �  n  8    �  �  �  |  i  U  =  $  	  �  �  �  �  w  U  +  �  �    &  �  �  �  �  �  �  �  z  s  l  c  X  M  B  7    �  �  ~  M  -  $    �  �  �  �  �  ]  9    �  �  �  Q  #  �  �  �  �  �  �  {  q  h  ^  T  K  A  8  .  $         �   �   �   �   �  f  i  k  k  g  a  W  H  5  !    �  �  �  �  s  N  (    �  ,    �  �  �  �  �  u  Z  E  1      �  �  �  �  �  �  �  	�  	�  	�  	o  	R  	1  		  �  �  x  0  �  �    �    ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  l  `  T  H  <  0  %  �  L  �  &  ^  s  x  k  C    �  #  �  �  �  �  <  X  	C  t    d  ?    �  �    A    �  �  �  �  ]  $  �  �  #  |  �  W  G  6  &        �  �  �  �  �  �  �  �  {  R    �  �  �  �      �  �  ~  0  �  P  �  *  �      �  �  7  
Q  �