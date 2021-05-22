CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?׮z�G�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�   max       PՆ
      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �+   max       =�      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @E�          p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�X    max       @vhQ��     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @Q�           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��1   max       >��H      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�#�   max       B,��      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�}C   max       B,��      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?:�   max       C�      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��F   max       C��      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         s      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�   max       P�f      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���m\��   max       ?�vȴ9X      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       >Kƨ      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @E�          p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���P    max       @vf�G�{     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q�           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�A           �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A-   max         A-      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��N;�5�   max       ?�oiDg8     �  N�         )  r         i               "      
         C   "   9      <   3                        
         ?   -         J   	      $                     k   E      4            O�LN��O�1`P��RO7��O���P�0�N=�6N~�Oo��NHNWO�K9NI��N��|O�'O�$PKڜOvN�P[n7O�^yPՆ
P,�O?q@N�ږO���OVV�N��O�O&%�Np�FOv`xN���P�+vP&,NxYN]PY�NW��O/MO9�N�OXr
O+�NS�O@��N�݄O�p|O<�;NTi�O��8N7�NN�%0N%	�Nrh&�+�o�49X���
���
��o;�o;�o;ě�<49X<e`B<u<�o<�o<�o<��
<�1<�1<�9X<�9X<�9X<�j<�j<ě�<ě�<���<���<���<�h<�h<��=o=o=o=t�=0 �=8Q�=8Q�=T��=aG�=ix�=�C�=�O�=�O�=�hs=��=��=���=���=���=��=���=�Q�=�����������������������������������������B=HUan��������znaSHB)5N��������t[B������� ���������!#-0<U_fggbUS<0#!�����5B[fdVK7)���int�����utiiiiiiiiii"!#%0530%#""""""""ss�����������������s}�������������}}}}}}�������(0)&"�������������������������������������������#*6BO[gkkfa[OB6#!���������������������������������������
#*-,&#
����iffmz������������zqi����� -.)�����7B[����#3-������NB7~}�����������������~-*+0<H\appna`\UPH</-��������������������^`eh�������������th^��������������')*59>?5)#JCNgt���������qfe[NJ������������������������������������������������

���PUalnqz~zvz{zna`WUPPmx���'/2)�����tm��������� ����������
 #/47/.#
_][_adgjnnnnja______����#4041382)����0..269?BFECB=6000000��������������������#0<ILRPI@<0#%#()15885)%%%%%%%%%%4//5;@FHTajlkia[NH;4)+-.353/))*)' �����������������������������������������
#+4870#
�������������� �������LNUZansona`ULLLLLLLLvz����������������|v
 $"#*-(#

#######genz����znggggggggggttv��������uttttttt�'�3�@�Y�e�m�r�l�e�Y�L�B�C�@�3�'����'�0�=�?�I�O�O�J�O�I�=�:�0�$��$�&�!�$�,�0����������
�����������������������������O�tĚįĲĭĔč�t�[�O�B�6�)�����)�O�6�B�O�[�r�x�w�t�h�[�B�>�8�<�6�)�&�)�/�6�Y�r����������ļ�����������h�f�\�T�V�Y�#�<�U�m�z�r��u�s�b�U�<��������������#���������������������������������������ѻ����������������������������������������-�:�<�3�2�;�?�:�-���� ���������!�-��������!�)�!�����������������������"�.�;�G�T�a�j�l�`�T�G�C�;�"�	�������"�f�s�}�����|�s�k�g�f�^�f�f�f�f�f�f�f�f�����ʾ׾ݾ���۾׾ʾ����������������������ʾ׾���ܾʾ�������v�t�u�y������àäìù����������ùìàÓÇÅÇÊÖÛà���������3�@�O�Y�C�6����ŭţŞŤŽ�ҿ����������������y�m�`�T�P�K�M�T�]�m�z���g�������������������s�g�A�6�4�8�9�A�N�g�(�5�?�A�>�A�<�E�A�(����������	���(�"�8�=�a�r�r�T�4�"�	������������������"��������)�/�0�)��������ñìû�������������������������������������������޿��������ĿѿѿٿѿĿ��������������������T�`�m�o�s�p�q�~�}�y�m�`�T�K�J�;�5�7�B�T�b�n�o�p�n�c�U�H�<�/�,�&�$�/�8�<�H�U�a�b�����������������������������������������y�������������y�`�T�G�;�1�0�;�G�T�_�j�y�5�A�N�Z�c�g�l�g�f�`�Z�N�A�5�3�(�%�%�(�5�����������������������������������������[�g�t¦²²ª¬¦�g�[�N�@�A�N�[�����������������z�s�h�g�e�d�f�g�s�u����������*�D�M�@�/����������������������׽���4�D�H�>�<�4�*���߽н����������ݽ����!�(�*�-�(����
��������������ûлԻлû��������������������������ܻ�������л����l�S�F�-�*�:�:�F�l���ܺ����!�+�!����������������������S�Z�`�l�y�����������������y�l�`�S�U�N�S�ֺ���������������ֺɺǺú��ɺκ�ƧƳ������ƾƳƧƤƣƧƧƧƧƧƧƧƧƧƧĚĦĳĿ������������������ĿĮĦĚĔēĚ�N�g�t¤�t�g�[�N�@�<�B�J�N¿����������������¿²«²¼¿¿¿¿¿¿�Ϲܹ���������������ܹϹù��ù�E7ECEPEVE\E`EeE\EPEMECE7E6E3E7E7E7E7E7E7EuE�E�E�E�E�E�E�E�E�E�EuEiEgEeEeEcEiEkEuD�D�D�D�D�D�D�D�D�D�D�D�D{DoDnDoDrDwD�D�������(����������������ּ߼��� ������ּʼ��������������ֺY�e�q�r�t�r�e�Y�N�P�Y�Y�Y�Y�Y�Y�Y�Y�Y�YǭǡǔǈǂǄǈǒǔǡǭǱǲǭǭǭǭǭǭǭ�����������������������������������������@�C�M�X�N�M�@�4�,�'�%�'�4�>�@�@�@�@�@�@ 9 F J / [ 1 8 F 3 Z � " H M  , > B / G U  7 N c W 3 E  O ` E H N h � _ b 5 * / Y j 3 j I / 8 i 5 / 1 h /  "  '  
  3  �  l  z  `  ;  #  �  �  g  
  �  .  �  	  �  T  W  �  �    ]    �  �  ^  �  2  �    (  �  �  N  �  �  �    �  �  \  �  �  �  �  �  S  T  �  n  m�o��1<�h>��H<��
<�j=�/<o<t�<���<�t�=D��<�t�<���=0 �=49X=�9X=aG�=���=t�=��T=��P=Y�<�`B=,1=,1=+=,1=H�9=�w=L��=�P=\=��-=,1=L��=�F=Y�=�C�=�Q�=y�#=� �=�E�=��=���=�1>8Q�>t�=��>J=�9X=���=��>O�B!P/B�B-�B�0B��B&ieBB	�=B%a6B �B*�B�?B��BD�Be�B!��BnB��B ��B�B�RB�gB�B��B�BT�B�CB	��BqB�Bc6B��B��B!��B�iB��B�B��B,��B%�oB�A�#�BY,B4cBQ�B�
B�hBwB B��Bz,BT]B �B��B!E�B��B%JB�aB�zB&D�B��B
�B%�9B ��B*=�B��B�qB;�B��B"?FB;�B;�B ��B��BU_B?�BϯB��BkwB�HB��B	ιB=7B�tB�MBQ�B��B!�|B��B6KB�HB�/B,��B%��B��A�}CB1�B9?B�fB��B�.BB�B;�B:�B�eB@B��B�(?��OB
}YA�cA�q1A�8�@�A^A��A�y2@���@gK~@_�8Aa��AB��AO�KAL=A�5�A�p�Al�A��bA��A���A��rA��Av$9Ai�A���Asa�Ai��A���A�'ZA�xA��HA�!�A.G A���@�s+@��U@]2�A�@E��B.�A�ddA��A��O?:�C��EC�C��A34aA N�?ᷜB�s@!�@� �?��B
��A�k�AڀAي�@�+A阔A�{�@�q�@dC�@d�>Aa'�AB&tAQ!AL��A�qA���AkdA��oA���A��AҀA�p�Av��Ag.�A�~�AsvAj��A�l`A�PA��,A��NA��A0�*A���@�M@��@]d�A�c@Go�B;A⌒A�=A�1U>��FC��:C��C��OA3A �P?��yB�m@#�!@��         )  s         i               "      
         D   #   :      <   4               	                  @   .         K   	      $                     l   E      4                     !   =         5                              /      -      G   +                           !      C   /         9                                                                        #                              #      %      9   )                           !      A   /         +                                                   N�?�N�+_OP�O�m�N�2sO{�O�^N=�6N~�N���N�UO�NI��N��|O�5DN�1�O���O	VP��O�^yP�1jP(IN�\%N�ږO)��OVV�N��O�N�ڥNp�FOv`xN���P�fP&,NxYN]P!ʯNW��O/MO9�N�OXr
O+�NS�O�N�݄OT�BO53XNTi�OM<rN7�NN�%0N%	�Nrh&  B  i  v  �    B  	�  �  �  �    i  s  �  �  �  �  �  �  i  �  O    �  �  �  �  6  �  �  �  �  �  �  �  �    �  Y    �    �  q  �  G  F  l  )    s  �  o  V���
��;o>Kƨ��o:�o=<j;�o;ě�<u<u<��<�o<�o<�j<ě�=#�
<�=+<�9X=�P<ě�=t�<ě�<�h<���<���<���=o<�h<��=o=+=o=t�=0 �=ix�=8Q�=T��=aG�=ix�=�C�=�O�=�O�=���=��=��=��w=���=�9X=��=���=�Q�=�����������������������������������������`YVXanz��������zna`1/15BN[grz}|wtg[NB61��������� �������� !%+0<IXabb]UI<0,%$ ����+6951%����int�����utiiiiiiiiii"!#%0530%#""""""""�������������������������������������������������������������������������������������������%'-6BO[`deec_[OB6/+%�����������������������������������������
#%((#
��klqz�������������znk����� -.)�����bp�������
"$�����pb�~������������������;229<HKUYUUJH<;;;;;;��������������������bdh�������������tkhb��������������')*59>?5)#JCNgt���������qfe[NJ������������������������������������������������

���PUalnqz~zvz{zna`WUPPtny���/1(�����zt��������� ����������
 #/47/.#
_][_adgjnnnnja______������-+/.+)����0..269?BFECB=6000000��������������������#0<ILRPI@<0#%#()15885)%%%%%%%%%%4//5;@FHTajlkia[NH;4)+-.353/))*)' �����������������������������������������������
"+0/,#
������������ �������LNUZansona`ULLLLLLLL��������������������
 $"#*-(#

#######genz����znggggggggggttv��������uttttttt�'�3�>�@�L�M�L�F�@�3�'�&� �%�'�'�'�'�'�'�0�2�=�I�L�M�I�F�=�0�%�(�$�0�0�0�0�0�0�0�����������������������������������������O�[�h�uĂąĄ�}�t�h�[�O�B�>�5�2�2�6�B�O�B�O�[�h�o�t�u�t�t�h�[�O�C�B�;�>�B�B�B�B�����������������������p�f�`�X�Y�f�r���#�<�K�Q�R�P�I�<�0��
������������������������������������������������������ѻ������������������������������������������!�"�-�2�:�:�/�-�!���� ��������������!�&�!������������������������.�3�;�G�Q�S�M�G�;�.�"������"�-�.�.�f�s�}�����|�s�k�g�f�^�f�f�f�f�f�f�f�f�����ʾ׾ݾ���۾׾ʾ����������������������ʾؾپپӾʾ������������~���������àìùý������ÿùìàÓÉÐÓÜàààà���������*�7�<�:�6�%������ŸŮŲ���ҿm�y����������������y�m�`�W�T�Q�U�`�g�m�����������������������s�Z�N�B�B�C�L�g���(�5�?�A�>�A�<�E�A�(����������	���(��/�:�H�R�b�g�g�L�4�"�����������������������)�.�/�(���������òíü�����������������������������������������޿��������ĿѿѿٿѿĿ��������������������T�`�g�o�i�h�m�t�r�m�^�T�G�E�=�>�=�G�K�T�b�n�o�p�n�c�U�H�<�/�,�&�$�/�8�<�H�U�a�b�����������������������������������������y�������������y�`�T�G�;�1�0�;�G�T�_�j�y�A�N�Y�Z�e�d�]�Z�N�A�6�5�(�(�(�(�5�>�A�A�����������������������������������������[�g�t¦²²ª¬¦�g�[�N�@�A�N�[�����������������z�s�h�g�e�d�f�g�s�u������������*�C�M�?���������������������������4�D�H�>�<�4�*���߽н����������ݽ����!�(�*�-�(����
��������������ûлԻлû����������������������������лܻ��������û����x�l�S�J�Q�x�����������!�+�!����������������������S�Z�`�l�y�����������������y�l�`�S�U�N�S�ֺ���������������ֺɺǺú��ɺκ�ƧƳ������ƾƳƧƤƣƧƧƧƧƧƧƧƧƧƧĚĦĳĿ������������������ĿĮĦĚĔēĚ�N�g�t¤�t�g�[�N�@�<�B�J�N¿����������������¿²«²¼¿¿¿¿¿¿�Ϲܹ�������������عϹȹù��ùȹ�E7ECEPEVE\E`EeE\EPEMECE7E6E3E7E7E7E7E7E7EuE�E�E�E�E�E�E�E�E�E�E�EEuEoElEkEmEuEuD�D�D�D�D�D�D�D�D�D�D�D�D{DpDsDxD{D�D�D�������(����������������ʼּ�����������ּʼ��������������ʺY�e�q�r�t�r�e�Y�N�P�Y�Y�Y�Y�Y�Y�Y�Y�Y�YǭǡǔǈǂǄǈǒǔǡǭǱǲǭǭǭǭǭǭǭ�����������������������������������������@�C�M�X�N�M�@�4�,�'�%�'�4�>�@�@�@�@�@�@ 3 7 0  ) . . F 3 / [ ' H M  . : 4 % G F  " N o W 3 E  O ` E J N h � W b 5 * / Y j 3 Y I ) 8 i : / 1 h /  �  �  �  7    �  �  `  ;    B    g  
     �  R  D  �  T  `  �  �    �    �  �    �  2  �  q  (  �  �  	  �  �  �    �  �  \  r  �  �  s  �  �  T  �  n  m  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  .  {  �  �  �  �        3  A  6  #    �      �    �  M  L  P  a  c  X  K  =  .        �  �  �  �  �  �  �  h    o  �    J  i  u  p  a  I  #  �  �  V  �  k  �  6  T  �  �  _    a  i  5  �      �  �  �  _  q    R  �      d  �  �  �      	  �  �  �  �  }  Q    �  �  W  �  {    �  �    1  ?  @  5  %    �  �  �  �  �  U    �  �  $    �  }  �  j  �  	*  	d  	�  	�  	�  	�  	�  	F  �  g  �  O  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  {  u  o  o  r  u  x  |  �  �  �  �  �  �  �  �  �  ~  z  w  s  t  �  �  �  �  �  �    *  R  u  �  �  �  �  v  h  U  B  3  "    �  �    7  m      	        	    �  �  �  �  �  �  �  �  �  �  �  �  �  �    <  Q  X  \  `  e  h  g  a  T  2  �  �  t    �  L  s  o  k  g  c  _  [  W  R  N  I  B  ;  4  -  &          �  �  �  �  �  �  �  �  |  a  D  *          �  �  �    �  �  �  �  �  �  �  �  �  �  r  R  ,  �  �  K  �  �  �  _  �  �  �  �  �  �  �    [  7    �  �  �  S  (  �  �  �  �  �  h  �  �  �  �  �  �  �  �  h    �  Z  �  �  �    ,  U    ?  e  |  �  �  �  y  b  B    �  �  0  �  l    �  �  o    h  �  �  �  �  �  c  3  �  �  �  F  �  �  !  �  
    �  i  e  a  \  V  M  C  7  +  %  (  0  -  "    �  �  o  )   �  N  }  �  �  �  �  �  �  _  !  �  �  �  �  f    �  j  �  �  M  F  0    �  �  �  �  ^  $  �  �  Z  -  �  �  ;  �  �  P  �  F  {  �  �  �          �  �  �  O    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  b  F  )     �  ~  �  �  �  �  �  �  �  �  �  �  �  �  t  =  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  n  P  /    �  q    �  -  �  �  �  �  �  �  �  �  s  b  P  =  *      �  �  �  �  �  6  +      �  �  �  �  �  �  n  E    �  �  �  M    �  F  �  �  �  �  �  �  }  i  R  :  $  	  �  �  T    �  g    �  �  �  �  �    o  `  P  A  2      �  �  �    Y  3    �  �  �  �  �  �  �  q  F    �  �  �  m  E    �  �  $  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  N    �  �  �  f    �  \  �  =  �  �  G  q  �  �  �  �  �  �  w  j  W  ;    �  �  �  �  I  �  �  �   �  �  �  �  �  �  �  ~  k  V  B  ,    �  �  �  �  ~  K     �  �  �  �  �  x  c  M  8  #    �  �  �  �  z  O  %  �  �  �  �  �  	      �  �  �  p  @    �  �  C  �  �  >  t  �   �  �  w  g  V  C  0       �  �  �  �  }  _  B    �  q    �  Y  =  8  0  #    	            �  �  �  �  z  <  �              �  �  �  �  �  h  0  �  �  I  �  �  #  �  �  �  �  y  r  k  `  I  3      �  �  �  `  3     �   �   �   W    �  �  �  o  E    �  �  f  $  �  �  F  �  �  �  Y  +  '  �  �  �  �  �  �  �  z  \  8    �  �  }  D  �  W  �  ,  �  q  l  h  d  _  Z  R  J  B  :  2  +  #          �  �  �  �  �  �  �  �  �  �  �  �  k  4  �  �  t  ,  �  �  �    ;  G  '    �  �  �  �  �  d  =    �  �  �  _  ,  �  r  �  s  E  �  �  )  @  E  :    �  �  G  �  M  �  �  �  
�  	�    �  h  j  d  U  9    �  �  O  �  �  �  R  t  �  
y  	U  �     �  )  '  %  "    �  �  �  �  �  �  �  �  r  9     �  �  �  �  �  �  �  	        �  �  �  �  �  E  �  �  >  �  �  R  �  s  g  Z  N  A  3  %      �  �  �  �  �  �  {  a  A        �  j  T  <  !    �  �  �  T    �  �  h  /  �  �  N  �  l  o  a  S  D  6  %    �  �  �  �  �  �  u  c  V  J  >  2  &  V  N  ;  '    �  �  �  �  R    �  \  �  �    �  *  �  