CDF       
      obs    0   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�(�\)      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       PN��      �  l   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �C�   max       =�      �  ,   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�33333   max       @E�Q�     �  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�fffff    max       @vfz�G�     �  'l   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @N�           `  .�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           �  /L   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >O�;      �  0   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B �|   max       B'�      �  0�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       B �m   max       B'@�      �  1�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?R��   max       C��      �  2L   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Q(J   max       C��      �  3   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          o      �  3�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          /      �  4�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          !      �  5L   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       O���      �  6   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�PH   max       ?�3���      �  6�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �C�   max       =�      �  7�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @@G�z�   max       @E�Q�     �  8L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�{    max       @vfz�G�     �  ?�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @M@           `  GL   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  G�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  Hl   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u��"    max       ?���vȴ:        I,                              #   '            f   6   *      )   n                        #   /         9   :                                       f      NTkN���N�.�N�aO���O��O�M���N2CWOwpgO�=iN��4O?PN�7~P?lxPN��O�$IOr�QO�8+P1�O0a#O|�oN�X�Ob2N��/O��O:k�OGR�P�rP!3O6�O�u�O�VzO�NI��NA��N$��O���OV�_N��oN"~fN�A�N��.N��N�t�Ob��N���NK��C���C��D���D���o�o:�o;��
<t�<49X<D��<u<�o<�C�<�1<�9X<�9X<ě�<ě�<���<�`B<�h<�h=o=+=+=C�=\)=t�=�P=��=�w=#�
=,1=,1=L��=T��=}�=}�=�\)=�\)=��w=� �=� �=�9X=��`=�=�.-/<HJHE><;/........V[bht�����tjh[VVVVVV#-/2<=@><2/#����������������������������������������BN[cmsrg[A5)634<FHPU[aaa_ZUMH><6����������������������������������������"<HU\dcUHBBD<:7#3116?BO[my����~hOB;3<8<CILUbklbbUTI<<<<<#/045<=<8/*#]Zalnnz����zna]]]]]]������
8BG<8
����(""%5Ng������t[JB5)(�������������������������������������%/<HUnz���znaH#UUZg������������tk[UKRQR[hpt}����ztqh[OK��������������������������������{{�������������������)55:75)���ljlz�������������znl�����������������������������������������6Oagkjc[OB6)����/HanqpjaU<#
��
)6<BCBA:6) ��������
������������"&#�����������������������������������������������

������������������������������������!")-,		��#)6BIOPO;62)%)44565)))556@A:50,))))))))) ##&/<?FHH@</*%$$# )5BGEB?5*)(JNS[fgktvvtgd[PNJJJJ[XYUaknz��}ztna[[[[���������	
��������������������������srvz�������zssssssssE\EiEnErEjEiEcE\EPEFEPE[E\E\E\E\E\E\E\E\�L�O�Y�[�b�`�Y�L�@�;�>�@�@�J�L�L�L�L�L�LD�D�D�EEEEEEED�D�D�D�D�D�D�D�D�D߹������������������������������N�Z�g�s���������������s�g�Z�N�A�<�9�@�N�T�a�m�u�������z�m�a�H�>�6�/����"�/�T�����������������������������w����������������������������������������������������������������������������������������������������������������ù÷ù�����޼����ʼؼ������ּʼ������������������r��������������������t�r�h�r�r�r�r�r�G�T�`�m�q�u�n�m�`�T�=�;�2�.�-�.�0�;�<�G�/�<�G�H�H�H�D�>�<�/�(�#�%�)�/�/�/�/�/�/�5�N�g��r�g�5�)�������������5�����$�0�I�J�3�����������ƽƳƧƢ��������2�:�<�4����̽Ľ��Ͻ���������(�5�A�A�:�9�5�(����������������(�*�A�L�N�N�G�C�?�5�(����������B�O�[�g�u�t�n�[�O�6�)����������6�B�:�F�S�_�f�e�_�Z�S�F�:�-�*�!��!�&�&�-�:����������������������ĿĚďĐĚĦĳĿ�̼r�����������������x�r�f�]�Y�O�Y�f�n�r�H�T�a�m�v���������z�m�a�T�B�/�+�/�;�E�H�f�r�j�q�q�n�f�e�Z�M�K�C�F�M�M�X�Z�e�f�f�n�zÇÓÎÍÇ�z�p�q�n�a�`�c�\�`�a�c�k�n������'�4�9�<�4��������޻����ÇÓàìùûúùìçàÓÇ�w�p�n�o�z�|Ç���׾�������׾ʾ����������������������z�����������������������z�m�o�y��|�n�z���������	�
�������������������������׹ܹ��'�3�L�W�a�g�a�Y�3�'������ܹӹܼ�'�4�8�:�:�9�4�'���ܻлǻлջܻ����f�s�������������������s�f�\�P�N�S�^�f�s��������������s�k�i�s�s�s�s�s�s�s�s����������������������������������¿����������¿µ²«²¼¿¿¿¿¿¿¿¿����-�F�_�w�f�-�!���������ֺȺɺԺ⺋�������ºɺϺ׺ֺɺ����������~�{�~�����0�<�G�I�T�I�<�8�0�-�'�+�0�0�0�0�0�0�0�0�{�{ŇňŇ�{�n�b�X�b�n�z�{�{�{�{�{�{�{�{ǈǔǕǔǈǃ��{�o�b�a�V�S�N�V�b�o�{ǁǈ�(�5�A�A�G�E�B�A�5�.�(� ����"�(�(�(�(�N�R�Z�a�g�g�g�b�Z�N�J�B�A�=�A�C�N�N�N�N���ûлܻ������ܻлû���������������DoD{D�D�D�D�D�D�D�D�D�D�D�D{DvDoDmDgDhDoEuE�E�E�E�E�E�E�E�E�E�E�EuEjEuEuEuEuEuEu���!�(�.�8�.�!������������� F c ( / @ 6 . Z ? < 0 Z ? T D , N 3 @ & 2 R > 9 e u C  J e  N @  ? O 2 { 1 C T d 3 D 6 . O Y  ?  �  �  �    H  (  	  X    `  �  V  �  h  a  �  �  T  �  w      �  i  �  �  �  _  �  }  F  K  !  v  �  <  B  �  �  W    �  �    �  �  ��`B��o<o;�`B<�9X<�`B<t�<o<D��=8Q�=P�`<��
=C�<�`B=��m=���=�%=�P=��>
=q=#�
=D��=<j=Y�=8Q�=aG�=aG�=�\)=��=}�=e`B=ě�=ȴ9=��=8Q�=aG�=u=�E�=��T=���=���=�v�=ě�=Ƨ�=�l�>O�;>�>+B�cB0�B}B��B�B7Ba�B��B'�B��B4�B'�B��B�bB�B��B"~�BdBmzB
�QBːBQ�B"iB �|BLhB+�B!d�B"�BE�BdB=�B|B?&B�gB��B#��B G�B �B�B�9BG Br�B�B��B��B��BO6BmB��BG�B"�B�.B�BJ�BD6B��B��B�`B@�B'@�B��B�B��B	$�B"��B��BA�B
��B�2B>B!�-B �mBX�BAB!��B"?�ByB�B}�BJ�B9�B��B|}B$zB @B?�B��B�WB@�BG�B��B	>_B�zB�jB=rBC�C��?��C�C�?R��A���A���A��uA�~QA�nA�E@��@�۬AfŲA�!�A��B�SA1(A���A��KA��@��AA��$@���A���A?F�A���@��*AʉfAP�A�!QA��4?�T@��eAD�AD+�A��A���@p�@_�A�C�A���B��A�g�A��@��RC��OC��A��C�֛?�aC�G�?Q(JA��*A�rA���A���A�IAѼ�@�:@�LAg�Aí$A���B��A10�A���A��YA׊1@��BA�o@�mA�x�A? YAǀ�@�]�Aʆ�AOA�agA���?��J@���AC��AD-�A��A��0@s��@��A�A�8BG�A�b�A���@��C���C��A
ԑ                     	         #   (            g   6   *      *   o                        $   /         :   ;            	                           f                        %                           /   /   '      #   '                           '   )      '   #               +                                                                                       !                                                                                       NTkNR}�N�.�Ncf<O � O���O�M���N2CWN���O��ZN��4O��NYdO���O�ăOJ��Or�QO���Ohl O�!OX�N��N�hN���O��O:k�N޼�OR4�O��ZO�!O���O�T�Oy��NI��NA��N$��O1?�OV�_N��oN"~fN��N��.N��N�t�O_F�N���NK��    7  '    f  X  L    p  (  �    �    
�  A  �  8  b  �  �  �  >  g  �  �  �  /  �    6  T  �  R  �  -  �  �  �  B  �  <  �  �  �  N  �  ��C��u�D���o;ě�;�`B:�o;��
<t�<�`B<u<u<�C�<��
=�C�=,1=C�<ě�<���=��<�h<��=+=�w=\)=+=C�=8Q�=aG�=49X='�=e`B=D��=49X=,1=L��=T��=�\)=}�=�\)=�\)=���=� �=� �=�9X=��=�=�.-/<HJHE><;/........_ght�����th________#-/2<=@><2/#����������������������������������������)5>NXafg[ND5)634<FHPU[aaa_ZUMH><6����������������������������������������')/3<HHQNH><2/''''''5237BO[kv~���whOB=65<8<CILUbklbbUTI<<<<<#//348</#a\anz��znaaaaaaaaaaa�������
!%%"
����58?Ngt������tg[NB<75�������������������������������������&/<HUnz���znaUH#ffgt������������tqjfLSRT[hot|����th[ROL���������������������������������������������������)15753)ljlz�������������znl����������������������������������������##()6BO[]ab`[OB96/)#	�
/<MU\dfaU</#	)6:@B>86)%���������������������� ! �����������������������������������������������

��������������������������������������  !�#)6BIOPO;62)%)44565)))556@A:50,)))))))))+%%&/<>EG<</++++++++)5BGEB?5*)(JNS[fgktvvtgd[PNJJJJ[XYUaknz��}ztna[[[[����������
�������������������������srvz�������zssssssssE\EiEnErEjEiEcE\EPEFEPE[E\E\E\E\E\E\E\E\�L�Y�Y�a�^�Y�L�@�?�@�A�F�L�L�L�L�L�L�L�LD�D�D�EEEEEEED�D�D�D�D�D�D�D�D�Dߺ������������������������N�Z�g�s�|��t�s�g�c�Z�P�N�E�A�H�N�N�N�N�;�H�T�a�m�r�w�z���z�m�a�T�H�>�;�4�2�6�;�����������������������������w����������������������������������������������������������������������������������������������� �����������������������������뼱���ʼּ����ּʼ��������������������r��������������������t�r�h�r�r�r�r�r�G�T�`�m�p�t�m�l�`�T�G�;�.�2�;�>�G�G�G�G�/�<�D�E�A�<�/�'�(�/�/�/�/�/�/�/�/�/�/�/�)�5�B�N�[�e�i�f�X�N�B�5�)���
����)���$�3�9�2�'�����������������������������(�1�1�(�#�������ݽ׽ݽ������(�5�A�A�:�9�5�(���������������(�A�J�M�L�F�B�>�5�(������������B�O�P�[�`�a�]�[�O�B�6�)�����$�)�6�B�:�F�S�_�e�d�_�Y�S�M�F�:�/�-�!�(�(�-�0�:ĳĿ��������������������ĿĦĚđĒĚĦĳ�f�r��������������r�f�c�^�f�f�f�f�f�f�T�a�l�m�z�}���z�r�m�a�Z�T�H�J�Q�T�T�T�T�Z�f�n�o�l�f�`�Z�P�M�F�H�M�P�Z�Z�Z�Z�Z�Z�n�zÇÓÎÍÇ�z�p�q�n�a�`�c�\�`�a�c�k�n������'�4�9�<�4��������޻����ÇÓàâìðõìáàÓÇÃ�z�w�w�zÅÇÇ�ʾ׾�����پ׾ʾ������������������������������������������������y������������������	�����������������������������'�3�@�N�W�[�V�F�@�3�'�����������'���'�4�7�8�6�4�/�'�����޻������f�s�����������������s�f�^�Z�S�R�W�a�f�s��������������s�k�i�s�s�s�s�s�s�s�s����������������������������������¿����������¿µ²«²¼¿¿¿¿¿¿¿¿�����!�-�:�F�S�b�S�R�F�-�!������������������ºɺϺ׺ֺɺ����������~�{�~�����0�<�G�I�T�I�<�8�0�-�'�+�0�0�0�0�0�0�0�0�{�{ŇňŇ�{�n�b�X�b�n�z�{�{�{�{�{�{�{�{�V�b�o�{ǂ�~�{�o�b�V�V�N�V�V�V�V�V�V�V�V�(�5�A�A�G�E�B�A�5�.�(� ����"�(�(�(�(�N�R�Z�a�g�g�g�b�Z�N�J�B�A�=�A�C�N�N�N�N���ûлܻ������ܻлû���������������DoD{D�D�D�D�D�D�D�D�D�D�D�D�D{DwDnDgDiDoEuE�E�E�E�E�E�E�E�E�E�E�EuEjEuEuEuEuEuEu���!�(�.�8�.�!������������� F U ( . ' > . Z ? - - Z A V 4  , 3 <  0 N 9 " G u C  . Z  ) :  ? O 2 ` 1 C T / 3 D 6 . O Y  ?  �  �  d    0  (  	  X  �  2  �    ~  �  �  �  �  #  �  U  �  �  �  �  �  �  �  �  �  &    �  �  v  �  <  �  �  �  W  �  �  �    �  �    >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�    )  8  G  B  9  0    
  �  �  �  �  �  �  r  \  F  0         .  8  A  M  [  e  j  m  q  u  v  s  b  L  1    �  �  '  $        �  �  �  �  �  s  M  #  �  �  �  @  �  �  L           (  +  ,  (  !      �  �  �  y  @    �  }  9    A  O  X  \  `  d  e  \  P  @  (    �  �  6  �    R   �    &  .  :  J  R  X  T  J  8     �  �  �  y  5  �  r  �  Q  L  J  I  E  A  <  6  /  '        �  �  �  �  �  d  9      "  ,  6  ?  I  S  V  U  U  T  S  S  Q  N  K  H  D  A  >  p  o  n  m  l  k  j  ^  N  >  .         �   �   �   �   �   �  �    j  �  �  �  �    #  '  #    �  �  >  �  r    �  �  �  �  �  �  v  _  ?    �  �  9  �  �  !  �  7  �  ?  �  e            �  �  �  �  �  �  �  �  �  �  �  y  m  a  U  �  �  �  �  �  �  l  J  (    �  �  z  B  
  �  �  r  3  �  �  �  �  �  �                            �  �  �  S  �  	  	�  	�  
K  
�  
�  
�  
�  
d  
  	�  	5  �  �  v  �   �  �  �  �      )  ;  A  ?  -    �  �  6  �  _  �  C  �  }  �  N  �  �  �  �  �  �  �  �  �  x  @  �  �  1  �  X  �    8  '       �  �  �  �  �  o  _  G  *    �  �  t  ]  L  <  [  a  W  E  1    �  �  �  �  J  	  �  }  �  Q    �  �  �  	5  
5  
�  �  �  _  �  �  �  �  �  \  �  W  
�  	�  �  �  �  !  �  �  �  �  �  �  �  �  �  s  Y  5    �  �  �  �  �  �  �  t  �  �  y  h  S  <  #    �  �  �  d  2  �  �  1      8  #  -  6  ;  >  >  =  :  4  )      �  �  �  �  �  M  �  q  �  �    (  A  W  d  e  ]  P  ?  %    �  �  y  F    �  +  �  �  �  �  �  �  �  �  x  g  T  @  %    �  �  |  '  �  b  �  �  �  �  �  Q    �  �  �  `  	  �  9  �  =  �  ;  �  ?  �  �  �  �  �  �  �  �  �  t  G    �  �  �  k  q  �  �  E  �    !  )  ,  /  .  %    �  �  �  C  �  �  �  F  �  �   �    H  W  i  }  �  �  �  �  �  �  t  H    �  y    �    _  �  �  �  �          �  �  �  �  y  C    �  �  <  �  1  ,  5  5  6  3  *      �  �  �  �  �  �  y  ^  9    �  �  �  �  �  "  A  R  S  H  !  �  �  E  �  z  �  w  �  �  �   �  
  j  �  �  �  {  w  ]  9    �  �  Z    �    *  ,  %  �  7  K  P  M  L  L  K  I  B  4    �  �  �  w  @     �  ^  �  �  z  q  h  `  W  N  D  :  /  %        �  �  �  �  �  �  -      �  �  �  �  �  �  �  �  q  ^  I  4      �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  N    �  v  )  �  �  1  F  w  }  s  {  �  �  }  e  B    �  �  h  <    �  �  �  6  �  �  �  �  �  �  �  {  b  A    �  �  �  N    �  U  (    B  8  /  &        �  �  �  �  �  �  �  �  �  z  p  f  \  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  q  g  ^  ~    0      �  �  �  �  �  n  K  (    �  �    T  '  �  �  �  �  �  �  �  s  ^  F  +  
  �  �  �  V  #  �  �  r  .  �  �  �  �  u  i  \  O  A  2  !    �  �  �  �  �  �  b  D  �  y  =    �  �  m  ;  �  �  l    �  :  �  \  �    y  �  K  $  �  �  �  h    �  R  �  6  w    e  <    
�  	?  _  T  �  �  q  b  _  b  f  h  a  J  2    �  �  �  ]    �  �  S  �  p  U  E  3      �  �  �  �  �  �  �  K    �  �  E  