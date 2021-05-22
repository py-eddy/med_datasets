CDF       
      obs    0   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ҟ�vȴ9      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N.�   max       Q~�      �  l   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �D��   max       =��      �  ,   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @F33333     �  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ۅ�Q�    max       @vk33334     �  'l   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @N�           `  .�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  /L   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >�9X      �  0   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��i   max       B/��      �  0�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��    max       B0?�      �  1�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >F�0   max       C�e      �  2L   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�h   max       C�w      �  3   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         #      �  3�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          U      �  4�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?      �  5L   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N.�   max       P�\�      �  6   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�{���m]   max       ?�?�      �  6�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �D��   max       >0 �      �  7�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @F33333     �  8L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vk33334     �  ?�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @M�           `  GL   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           �  G�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  Hl   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�kP��{�   max       ?�>BZ�c         I,      (   ,            �   8                     P   (                        >   )         $   )   c      b   K      	              #               �         N��O�'ZO�ЀN5DO�;�N�5�Q~�P.��N�`�N�hNF�O���N+�O�P4!O��KO��EN�kN�O	qOnݷOhi�P�wOXP	�UO1/N��SO\.�P���O�'�O�%�O���PM�]O���O!�N��N.�O4�Ni�PR(WO,�;O�o>N�tN�.xO��
O���N�NӔe�D���t��t��t��o�D���o;D��;�o<49X<T��<�1<�9X<�9X<���<���<���<���<�/<�<�<�<�<��=o=��=�w=#�
='�=,1=0 �=8Q�=<j=<j=@�=H�9=H�9=q��=u=}�=�O�=�O�=�\)=���=�E�=�v�=�v�=��()56;95)#/<HUaiiaUH</#�������()4/)��������������������������"/;BHGBBA>>;/��������������������in�����)O\ZQ5����ri�����#<IPM<1�������
###
������[TUZ[[htwzywtjh[[[[[.//7<HPLH></........{~�����������������{-566BMMEB>66--------���������������������������
&/:95-���������
$%#�������sxpt�������������yus��������������������}z��������������}}}}�����������
�����ZUV^git��������thb\Zffkt������������xtpf��������������������lfz��������������{ulbbet��������������kbXY_ahmz���������zmaX#%/03<IKIIA<0#���� 

 ���wvz���������������}wRPVdnz���������zn^VR�������������������)( �������������)4HI2������������� �������()*6BLO[chnqkh\[OB6(:6<BKOVY[hpeab`[UOB:)#)*6?96+*))))))))))���� "$�����
!#)+#
���g[NB9-,/7B[g������������������������������%-11)����*'+/<HMTUXUH=<;/****���������������������������		������������$)/54)����967BDNOONB9999999999�������


��������t�o�l�r�t�z������ ������������������������������Ҽ�ݼּȼ������������������������'�3�6�6�5�3�'�"����!�'�'�'�'�'�'�'�'�H�T�a�m�z������z�m�a�H�;�"��	����H�����żʼռмʼ����������������������������
���<�U�c�c�I�#�
����ĳ�t�Z�Y�tĿ�廅���û˻ǻ�������������l�S�F�C�K�_�l���������������z�y�x�m�c�m�r�y�������������������	������	�������������������������������������������������������������{ǈǔǛǨǭǟǔǎǈ�{�o�b�_�b�c�g�j�n�{�����������������������ÓàäìóùüùìçàÓÇÀ�~ÁÇÇÍÓ���)�B�X�e�i�f�^�N�A�5�)���������������	�"�1�2�+�"��	�����Ӿʾľʾ׾ܾ�Z�s������������������f�Z�P�E�=�=�A�M�Z��'�)�*�'����������������G�T�\�`�a�e�e�`�T�G�;�;�0�/�;�?�G�G�G�G�f�s�������������������s�f�Z�W�Z�[�d�f�������������������������r�f�Z�]�f�r��A�M�Z�d�f�^�Z�O�M�A�4�(������ �(�A�Ľ۽����4�M�B�4��н�������������������������������������ùðåÞÞàìù�����	��(�-�-�*�"�&��	� ����������������ŭŹ������������������������ŹŬŪũŬŭ�'�4�?�@�M�T�M�I�F�@�4�2�'�#���'�'�'�'EiE�E�E�E�E�E�E�E�E�E�E�E�EuEiEbE\EXEaEi�g�s�����������������b�A����"�(�>�N�g���ܹ������������ܹϹù�����������������#�<�L�T�V�L�L�H�<�5�/�#�
���������
��лܻ���
��
�����л˻������������û��������������������������t�s�z�������������������������������������y�p�y�����#�(�1�<�>�A�<�7�/�#���
�	���
����������'�)�-�'���������߻�S�`�l�v�v�l�`�[�S�S�S�S�S�S�S�S�S�S�S�S��������������������������|�}����������ǡǪǭǰǭǭǡǔǋǎǔǖǡǡǡǡǡǡǡǡ�������������������)�6�=�F�H�A���������ɺкֺ������ֺϺɺ������������������������������������~�r�a�d�d�a�e�p�~������"�&� ����������������������������������ݽֽݽ���������D�D�D�D�D�D�D�D�D�D�D�D{DoDkDiDoD{D�D�D���������������������ƧƎƁ�|ƀƎƜƧƳ���uƁƎƏƎƂƁ�u�q�p�u�u�u�u�u�u�u�u�u�u�/�<�C�H�N�U�W�U�K�H�<�7�/�#�!��#�#�/�/ 5 D ; ^ J  L E v 5 c I L 9  = ( L . D + 5 p  H U C p    * - I N { L 1 > 7 & N 2 K $ R < 3  �     �  }  �  �  	�    �  �  s    Z  \  �  `        ^  �  �  P  �  �  �  �  *  �  �  �  2  r  V    1  *  �  �  �  m  �  �  �    �  -  �o<��=\)�o<���;��
>%�T=q��<#�
<e`B<�C�=<j<���=P�`=�
==��=49X=+=C�=H�9=u=e`B=aG�=�j=��=L��=L��=���=��>\)=�7L>n�=���=}�=e`B=�o=T��=��
=�7L>�9X=�^5=���=���=�v�>`A�=�=Ƨ�=��B��B��B|�B!�A��iB"V�B�B#zoB��B�_B��B��B=B!��B��B_�B%DB!��B�gB��BlB�B"R	B�B3A��,B%��B��B[�BXgB��B"�B�LB,�
B�UB��B/��BQ�BGLB	(�BJ�B��B��B<�B��B�B��B�B�|B��BB�B!9tA�� B"@�B��B#@B��B�eB1aB�B��B"@ B=�B��BFB!��B�	B��B>{B��B!@�B=�B
�.B &WB%�+BA1BT�BA�B��B>�B��B,�B�BB�B0?�B<�B9RB	6rB:�B�EB��B?B�BE�B�B��A��)A��@��?���A�؅@�i�A��@�$�An18A��7A��Bcv@X"&A��&A��\AY��ACda@�<�Ae��AD�u@�!A:$OA+�uA�mA��)A��)@�c�C�eA���>F�0A��M@��A�F�A!W�A�f�@�uA��Ar��B��A�̮@25@BA2��A0@C���B_�B*�A�O]A���A҂�@��?�AZA�N@�vYA�z�@�
An�CA��A�S�B��@Y�"Aʭ�A���A[$AC �@�ڗAf��AD�*@��ZA:�*A$-AΏ�A���A��@̈́C�wA��>�hA|@��A�N�AmA���@�(A��Ar��B�`AӅ@0	�@�tA2�A0�C��FB��B?�A�'�      )   -         	   �   9                     Q   )                        >   *         $   *   c      c   L      
              #               �                        !      U   /                     )      #                  3      )            3   !   !   #   -                     /      !                                       ?                        #      #                  3      !            /            '                           !                  N��N�%�O2kN5DOo��N�5�P�\�O3ªN�`�N�hNF�O���N+�O�O׺O�8MO��EN�kN�N�
�OnݷO[�P�wO��O��dO��N��OO\.�PnxO��O#�`O��P�O���O!�Nv��N.�O4�Ni�O�V@O,�;O�o>N�tNH��O_
O��N�NӔe  �  �  t  �     !  j  B    i  �  H    l  �  g  �  w  %  �  �  w    	�    a  C  �    g  &    �     �  o  �  �  l  �  �    E  
  c  `  2  o�D��;ě�<o�t���o�D��=}�=o;�o<49X<T��<�1<�9X<�j=L��<�`B<���<���<�/=C�<�<��<�=#�
=�P=�w='�=#�
=0 �=}�=]/=�7L=}�=<j=@�=Y�=H�9=q��=u>0 �=�O�=�O�=�\)=��=�/=��=�v�=��()56;95) #$/<EHPTHH</)$#    ������������������������������
"/;AB?<=;7/"!���������������������������)384������������	

���������
###
������[TUZ[[htwzywtjh[[[[[.//7<HPLH></........{~�����������������{-566BMMEB>66--------���������������������������
##!
�����������
#$!	�����sxpt�������������yus��������������������}z��������������}}}}�����������������ZUV^git��������thb\Zfglt�����������ztqlf��������������������vuz���������������zvecdht�������������oe`abjmz����������zme`#(0<GG?<0#���� 

 ���xx}���������������~x\XX\anz���������zne\��������������������������������������)5@B@2)������������ �������()*6BLO[chnqkh\[OB6(=;@BOQ[\^][POB======)#)*6?96+*))))))))))���� "$�����
!#)+#
:9<BN[gt������tg[NB:�������������������������%-11)����*'+/<HMTUXUH=<;/****�����������������������������	
	�����������#).30)	����967BDNOONB9999999999�������


��������t�o�l�r�t�z���������������������������������ʼμμʼʼļ������������������������'�3�6�6�5�3�'�"����!�'�'�'�'�'�'�'�'�H�T�a�m�o�v�y�u�m�a�T�H�;�3�"��!�"�1�H�����żʼռмʼ����������������������������������;�F�L�?�0�����ĿğČČğ���ػ��������������������x�l�j�a�l�n�x�{�����������������z�y�x�m�c�m�r�y�������������������	������	�������������������������������������������������������������{ǈǔǛǨǭǟǔǎǈ�{�o�b�_�b�c�g�j�n�{�����������������������ÓàâìòùûøìæàÓÇÁ�~ÂÇÈÑÓ�)�5�B�N�[�^�]�W�N�B�5�)�����������)����	��"�0�0�)�"��	�������־ʾ׾��Z�s������������������f�Z�P�E�=�=�A�M�Z��'�)�*�'����������������G�T�\�`�a�e�e�`�T�G�;�;�0�/�;�?�G�G�G�G�s�����������������s�f�_�`�f�j�s�s�s�s�������������������������r�f�Z�]�f�r��A�M�Z�c�e�]�Z�N�A�4�(������"�(�4�A�Ľ۽����4�M�B�4��н���������������������������������������÷ìââåìù�������	��%�)�)�%�!��	������������������Źź��������������������ŹŸŭŭūūŭŹ�'�4�;�@�J�E�@�4�'�%�� �'�'�'�'�'�'�'�'EiE�E�E�E�E�E�E�E�E�E�E�E�EuEiEbE\EXEaEi�g�s�����������������s�A�5���$�*�@�N�g�����ùܹ������ܹϹù��������������#�/�<�H�K�N�H�<�0�/�#��
����
���#�ûлܻ������������л����������������������������������������������~���������������������������������������y�p�y�����#�(�1�<�>�A�<�7�/�#���
�	���
������������������������S�`�l�v�v�l�`�[�S�S�S�S�S�S�S�S�S�S�S�S��������������������������|�}����������ǡǪǭǰǭǭǡǔǋǎǔǖǡǡǡǡǡǡǡǡ����&�/�4�5�3�)���������������������ɺкֺ������ֺϺɺ������������������������������������~�r�a�d�d�a�e�p�~������"�&� �������������������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DoDmDoDtD{D���������������������ƧƎƁ�~ƁƎƞƧƳ���uƁƎƏƎƂƁ�u�q�p�u�u�u�u�u�u�u�u�u�u�/�<�C�H�N�U�W�U�K�H�<�7�/�#�!��#�#�/�/ 5 + 2 ^ @  \ < v 5 c I L 9  4 ( L . 7 + 3 p  ; K @ p   6 ' * I N E L 1 >  & N 2 $ # Q < 3  �  �  ~  }  �  �  �  p  �  �  s    Z  N  �          �  �  �  P  2    f  �  *  �  X  ]  |  �  V    �  *  �  �  �  m  �  �  S  �  �  -  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  �  �  �  �  �  �  �  �  �  �  �  �  �  w  j  X  D  0      �  �  -  T  y  �  �  �  �  �  �  y  J    �  �  '  Q  -   �  ,  �  �    9  V  m  t  m  N    �  }  #  �  t    �  �    �  �  �  �  �  �  �  v  f  K  /      �  �  �  �  �  �  �  �  �  �                �  �  �  �  �  _    �  <  �    !      	  �  �  �  �  �  �  �    ^  -  �  �  �  S  +    	+  	�  
h  
�    8  P  g  a  1  
�  
o  	�  	�  �    �    f   �  �    Z  �  �      %  4  ?  A  ;  +  	  �  j    �    1      �  �  �  �  �  �  �  �  �  �  o  Y  =  !  	    �  �  i  c  \  V  P  J  D  @  >  ;  9  7  4  0  )  #          �  �  �  �  �  �  o  X  A  *        	    �  �  �  �  �  H  B  :  .    	  �  �  �  �  Y  +  �  �  �  @  �  �  u  �    �  �  �  �  �  �  �  �  �  �  �  z  l  Z  I  7  %      l  h  U  <       �  �  �  �  �  s  1  �  y  �  |  �  l  �  �    I  k  �  �  �  �  o  >    �  o  �  }  �  .    �   �  Z  f  d  `  >    �  �  �  G    �  r    �    |  �  h    �  �  �  �  �  �  �  �  z  m  _  Q  D  4    �  �  �  h   �  w  q  j  c  Z  Q  I  A  9  4  0  +  %                %      
  �  �  �  �  �  �  �  �  �  i  S  >  )     �   �  b  p  {  �  �  �  }  r  h  Y  F  $  �  �  w  #  �  �  9  �  �  �  �  �  g  ?    �  �  U    �  �  7  �  o  D    �  �  v  w  s  i  Z  I  3      �  �  �  �  d  5  �  �    �      �  �  �  �  u  O  E    �  �  �  u  J        �  v    	�  	�  	�  	�  	�  	�  	t  	D  	  �  �  *  �  N  �  %  {  �  �  �  �  �  �     �  �  �  W  ]  N  9  "     �  �  S    �  9  �  U  \  [  N  8      �  �  �  �  �  �  {  \  3    �  �  �  $  4  A  B  B  A  =  8  ,      �  �  �  �  �  e  @  �  �  �  �  �  �  �  �  `  '  �  �  T    �  �  �  o    �    �        �  �  �  �  �  m  =    �  �  z  C  
  �  �  :  �  �  (  O  e  g  b  K  &  �  �  J  �  H  
�  	�  	  %  �  �  f  �  �  �  �  �  �      #  %      �  �  �  �  ^  0    �  �  �  0  g  ~  z  h  F    �  s    
�  	�  	3  a  D  �  o  �  ;  �  �  �  �  �  �  �  d  $  �  {    �  N  �  z  �  �                 
    �  �  �  w  G    �  �  �  ?   �   �  �  �  �  �  �  �  �  �  v  _  G  0      �  �  �  �  4   �  `  =    +  f  o  f  U  =    �  �  �  s  I  "    �  �  �  �  �  �  �  �  �  �  �  q  X  @  '     �   �   �   �   �   �   �  �  �  ~  t  i  ]  L  6    �  �  �  |  H    �  {    y  �  l  S  :  %      �  �  �  �  �  �  �  �  o  M  +  �  �  U  G  �  c  �  u  &  u  �  �  B  �  �  �  �  X  �  R  (  �  �  �  �  �  �  t  \  B  $    �  �  �  Z    �  �  [    �  �    
  �  �  �  �  �  �  f  6    �  �  [  !  �  �  �  I  �  E  7  +        �  �  �  �  �  w  R  *    �  �  �  l  C  �  �    	  
     �  �  �  �  ]  7    �  �  p  7  �  �  �  �  "  P  a  @    �  t  �  ^  �    8  W  -  9  �  �  �  �  _  _  W  E  .    �  �  �  u  P  +     �  �  b    �  �   �  2       �  �  �  �  �  s  S  6      �  �  �  �  �  g  L  o  Y  G  <  ,    �  �  �  �  Z  1    �  �  h  /  �  �  p