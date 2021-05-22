CDF       
      obs    1   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��x���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P���      �  p   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       :�o   max       >hs      �  4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E���Q�     �  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       @
=p��    max       @vh          �  '�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @M�           d  /H   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��`          �  /�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       <u   max       >dZ      �  0p   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,��      �  14   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�k   max       B,�      �  1�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @5a   max       C��      �  2�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @6rw   max       C�`      �  3�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  4D   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  5   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  5�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P}��      �  6�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�D��*1   max       ?��t�j      �  7T   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ;o   max       >hs      �  8   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E���Q�     �  8�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       @         max       @vh          �  @�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @M�           d  H,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�           �  H�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         DT   max         DT      �  IT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?x�t�j~�   max       ?�Ov_�     P  J               
   R   r      	   $      &   K   *      &      
         �   	            "         '   f   %      	   �      	            5      "            r   	   =   #OHb`O��N-hN���N�� PX�Pq��N��Nʥ9P	�N��O�P�1�OՏ�P3	�O9q�N�x�N�H�N�r�NT�{P���O54N�N�N7M��P+��N5 �N���Pc��P��O�Q�N��.N�ջO�	�N�b�N�#�O�nM���N���O��[O)9�O�C�N�3�N�GRN�F�P`M�6O��BO-:�o;�o;�o<#�
<T��<e`B<u<u<u<�o<�o<�C�<�C�<�C�<�1<�1<�9X<�9X<�j<ě�<ě�<ě�<�/<�`B<�`B<�<��=�P=�P=�P=�P=��=8Q�=H�9=L��=aG�=q��=y�#=�o=�o=��=��-=��-=�9X=�Q�=���=�`B>I�>hs��������
�������������� ������prty�����tpppppppppp������������������������������


��������)BH7:/'%�����������):;;5����)+00)��������������������@;<EO[t{wy���~uh[XS@}|�����������}}}}}}++0<HUcntxwnhaUHC=1+D?>MU`���������paUKD��������������������(���������������������������XZ[got}��������tg[XX#*0<DIJLIF=<0-#xx|�����������xxxxxx������������������������5[x��tqeN)�����������������������fhsttv����thffffffffTTWamz{}zmaTTTTTTTTT��������������������������
(<HI</#
���������������������������������������������)5GMNH?)�����������#.5@EFB<#
���OKJJKMamz}����znaUO���
! #$#
���������������������������������

������~�����������������~~#'.0310+#��������������������"')6786.)""""""""""	"/0//&"	�������������\Zajmz�������{ymhbb\{x|����������������{�������������������������������������������
 #'(##

�����igebcnz����������zqimnnz}}{zonmmmmmmmmmmnz���������������zqn����������		����ֻ������ûллܻۻлŻ�������������������Ź��������������������ŹűŭŦŧŭŵŹŹ�������û̻û��������������������������������������������������������������������)�4�6�6�B�O�Z�[�h�i�h�[�O�D�B�<�6�)��)���л������޻߻�ܻû��x�_�:�*�"�:����Ŀ������I�b�o�w�x�v�n�U�0����������ļĿ���$�*�1�*����
����������������'�+�4�>�5�4�'�"����������������(�A�Q�\�h�f�Z�4���нĽ��������н��s�����������������y�s�m�j�s�s�s�s�s�s��������������������������������(�5�N���������������s�N�5����������(�������������
������������������������N�s�����������������������|�Z�C�5�.�@�Nù����������������üùìÓÊÈÓÙàðù�y������������������������z�y�r�r�w�y�y�������������������������r�r�o�r�z���zÇÓÛÙÓÇÇ�z�u�n�k�n�n�z�z�z�z�z�z�������������������������������������������������)�M�c�f�[�O�6�����������óó�ſѿݿ���������������ݿڿӿѿ˿Ѽ���� �������������������ĿĿѿܿڿҿѿĿ������ĿĿĿĿĿĿĿĿļ���������������~����������������������T�`�m�r���������y�Y�?�.�
�������"�P�T�ּ����ּмʼ������ȼʼμּּּּּֿݿ������������ݿӿѿɿѿۿݿݿݿݿݿ�Ƴ�����$�A�;�/�������Ƴ�g�a�e�uƁƚƧƳ����)�7�>�E�B�5�)�������������������z���������������������������z�u�j�h�m�z�������������������������������������a�m�y�z���z�o�m�a�_�X�W�a�a�a�a�a�a�a�aD�D�D�D�D�D�D�D�D�D�D�D�D{DcDWDYD]DoD�D��H�Q�U�\�a�l�a�U�I�H�<�/�)�'�,�/�<�=�H�H�����!�%�+�!��������������������������Ľнݽ�ݽҽ̽Ľ����������~���}���ݿ�����������ݿڿݿݿݿݿݿݿݿݿݿ��"�(�/�;�=�A�=�;�/�.�"������"�"�"�"���������
��"�"������������¹µ��������������������������ŹŭŧŠŠŰŹ���������������������������������������������������������������z�t�w���������l�x�y�������������������������x�l�g�k�lǡǪǭǶǴǭǪǡǔǈǇǃǈǈǔǞǡǡǡǡ�����ֺ�� ��	������ֺ�����������������������ܻػܻ޻����������r�y�������������r�f�Y�M�I�@�<�6�G�Y�rEuE�E�E�E�E�E�E�E�E�E�E�E�E�E�E~EzEuEtEu # 3 O U [ Z D D U \ ; 1 $ J ] 7 \ " ) * ] U ` F W 5 v 5 [ 2 & ; 3 L 8 6 7 < J / B N < u ' , m 2 1    �  A  ?  5  �    I  =      �  2  ~  �  �  �       �  _  K  4  a  �  !  $  y  �  I  �  "  �  �  {  �  �  ]  "  �  �  �    �     �  a      x<�1<���<u=+<�9X=\>o<���<ě�=T��<�9X=aG�=�^5=q��=49X=m�h<�h<��=t�<��>N�=+<��=o<�=�%=t�=49X=���>V=��P=P�`=]/>dZ=��=�%=�\)=��=�t�=�h=���=�G�=�1=���=�
=>Z�=���>J��>49XB#+eB�B��B��B^B�kB+rB�B"�B̈B6BmB�B��B��B!�SB	�B%��B
��B'�B�!B�HBQ�A�=B#4MB�B!FBTcB^B��B��B�NBvB��B�`B%"�B+yBP�A���B��A��B� B�nB,��B�B�B*�BdB�TB#?�BW�B�WB�B�9B�NB@B��B"8�BADB?B6�BB�B��B;�B"@iB	��B&�B
��B?�B��B��By0A�0�B#H�BƢB!5^B?�BApB8�BU�B;zB�B?�B�8B%=B+<�BC�A�kB��A��BB�B�AB,�B,�B?QB@MB��B��@���A�g@�A�A�wzA���@��bA�&A�Rg@�S8A2uAE�9A�1�A���A��dA��~A��8An�@�:A�"qAЅA�6�A�[�@�lAy��@�T;Adj�@��A~E�BH�A�hA��A���A�.MC��[Aþ�A	�A"\�ALA�7&A��]A��A��:AFG@�ֲB�d@5a@��q@޽C��@��%A�.5@��AЀ3AؒQ@��A�^_A�D�@�9A2�AF#aA�~�A���A��6A��ḀAm;@��NAȊ�AЃ&A���A0@��Ay�@�:Aj�0A ��A~��BJA�u&A�r�A��A��"C��A�PA�A#�A~�JA��A� �A��
A��[AF��@� B� @6rw@��@� FC�`               
   R   r      
   %      '   K   +      &      
         �   
            "         '   g   &      
   �      	         	   5      "            r   
   >   #                  7   1         +         7   %   1                  =               -         7   '            #                  !      #            %                           %   %         )         3      #                  #               +         7                                       #                     O:��N�\�N-hN54N�� O�v�O�~N��Nʥ9O��UN��N���P}��O�B+O�FBN�5N[M�N�H�N�r�NT�{O�
N]��N�N�N7M��PrN5 �N���Pc��O�=O�f�N��.N�ջO^�N�d�N�#�O�nM���N���O�aO)9�O�C�N�3�N�GRN�F�O�Y�M�6O��O-  
  �  `  �  �  	  
�  �  V  �  �  T  �  �     �  �  V  �  �  �  d  8  �  �  �  �  �  �  �  `  �  E    �  �  �  m    �    �  |  3  �  \  ;  
.  �;o;ě�;�o<�C�<T��='�=aG�<u<u<��
<�o=t�<���<��
<�h=��<ě�<�9X<�j<ě�=\<�/<�/<�`B<�`B=o<��=�P=�P=��w=�w=��=8Q�=�h=Y�=aG�=q��=y�#=�o=��=��=��-=��-=�9X=�Q�>�=�`B>V>hs���������
������������		���������prty�����tpppppppppp������������������������������


��������(,))%���������� &+)$����)+00)��������������������B>?BO[hnsv��~yth^YOB}|�����������}}}}}};426<HOU[XUIH<;;;;;;ECPa����������naVPKE��������������������������������������������������]agt|������|tg]]]]]]#*0<DIJLIF=<0-#xx|�����������xxxxxx��������������������(5BN\jjdb[NB5��������������������fhsttv����thffffffffTTWamz{}zmaTTTTTTTTT�������������������������
'3<HHE</#
��������������������������������������������)5GMNH?)����������
#-131-#
���PLKKLMUWaiz���~znaUP���
! #$#
��������������������������������

��������������������������#'.0310+#��������������������"')6786.)""""""""""	"/0//&"	������������\Zajmz�������{ymhbb\{x|����������������{�������������������������������������������
 #'(##

�����tsrpsz������������ztmnnz}}{zonmmmmmmmmmmrpqz��������������zr����������		����ֻ������ûλϻлڻڻлû�����������������Ź����������������ŹŵŭŨŪŭŸŹŹŹŹ�������û̻û��������������������������������������������������������������������)�4�6�6�B�O�Z�[�h�i�h�[�O�D�B�<�6�)��)�����лػ߻�ݻû��������l�_�M�J�S�a�����
�#�<�I�U�`�b�a�Y�U�I�0��
� ���������
���$�*�1�*����
����������������'�+�4�>�5�4�'�"����������������(�2�K�W�\�Z�M�4���нĽ����Ľн޽��s�����������������y�s�m�j�s�s�s�s�s�s���������	����������������������������5�N�������������s�N�5�!�	��������(�5�������������������������������������s�����������������������������g�Z�X�Z�sùù��������ùìàÖààì÷ùùùùùù�y���������������������y�t�t�y�y�y�y�y�y�������������������������r�r�o�r�z���zÇÓÛÙÓÇÇ�z�u�n�k�n�n�z�z�z�z�z�z������������������������������������������)�6�B�O�S�S�O�B�6��������������������������������߿������������� �������������������ĿĿѿܿڿҿѿĿ������ĿĿĿĿĿĿĿĿļ���������������~����������������������T�`�m���������y�`�W�<�.�"��������"�T�ּ����ּмʼ������ȼʼμּּּּּֿݿ������������ݿӿѿɿѿۿݿݿݿݿݿ�Ƴ�����$�A�;�/�������Ƴ�g�a�e�uƁƚƧƳ�����*�.�.�)�&��������������������z���������������������������z�v�l�i�o�z�������������������������������������a�m�y�z���z�o�m�a�_�X�W�a�a�a�a�a�a�a�aD�D�D�D�D�D�D�D�D�D�D�D�D�DzDoDsD{D�D�D��H�J�U�X�U�U�H�F�<�/�+�)�-�/�<�F�H�H�H�H�����!�%�+�!��������������������������Ľнݽ�ݽҽ̽Ľ����������~���}���ݿ�����������ݿڿݿݿݿݿݿݿݿݿݿ��"�(�/�;�=�A�=�;�/�.�"������"�"�"�"���������
������
��������������������������������������ŹŭŧŠŠŰŹ���������������������������������������������������������������z�t�w���������l�x�y�������������������������x�l�g�k�lǡǪǭǶǴǭǪǡǔǈǇǃǈǈǔǞǡǡǡǡ������������� �������ֺɺ���������������������ܻػܻ޻����������Y�f�r�w����������}�r�f�Y�M�J�B�>�8�I�YEuE�E�E�E�E�E�E�E�E�E�E�E�E�E�E~EzEuEtEu ' ) O 6 [ N 3 D U T ; "  F M 6 k " ) * , + ` F W 7 v 5 [ $ " ; 3 H > 6 7 < J  B N < u ' 1 m / 1    �    ?  F  �  1    =    s  �  �  �  �  �  �  �     �  _    c  a  �  !  �  y  �  I      �  �  �  �  �  ]  "  �  3  �    �     �  }      x  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT  DT    
    �  �  �  �  �  �  �  �  w  T  .    �  �  I  �  �  t  �  �  �  �  �  �  �  �  u  _  D  $  �  �  �  �  ]    j  `  i  r  y  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    0  R  �  �  �  �  �  �  �  �  �  �  A  �  �  T  �  �  @  �  �  �  �  �  �  �  �  u  \  D  ,    �  �  �  �  �  �  u  %  ,  m  �  �  �  	  	  �  �  �  [    �  ?  �  �  	  �    V  	  	�  	�  
3  
x  
�  
�  
�  
�  
�  
�  
�  
h  	�  	!  3    t  f  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  V  O  G  J  N  I  @  ;  7  5  3  8  F  V  j  {  �  �  �  �  N  �  �  �  l  R  T  U  I  *    �  �  [    �  \    �    �  �  w  m  b  W  K  >  .      �  �  �  �  �  x  Z  =    �  >  w  �  �  �    6  D  P  Q  <    �  �  =  �  {  �  �  �  �  �  �  �  �  �  �  o  /  �  r    �  /  �  t    �   �  T  }  �  x  l  T  2  
  �  �  g    �  x  &  �  j  �    [    �  �  �             �  �  �  �  �  �  v  C  $    �  �  �  "  @  Z  o    �  �  �  �  k  I    �  �  B  �  �  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    F  �  �  V  H  9  9  :  ;  ;  5  ,    
  �  �  �  �  �  g  I  2    �  ~  o  _  H  .    �  �  �  i  -  �  �  m  ,  �  �  �  Q  �  �  �  �  �  �  �  �  }  t  l  c  W  L  8    �  �  �  �    �    T  �  g  �  �  �  �  �  >  �    *    
�  	]  W  +  -  #    &  6  I  ^  X  G  1      �  �  �  X  "  �  �  �  8  @  G  N  V  Y  P  G  =  4  .  +  (  $  !  !  !  !  !  !  �  �  �  �  �  �  z  p  f  [  P  B  5  (    	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  k  ]  N  @  2  $  �  �  �  �  �  p  @    �  �  �  q  M    �  �  >  �  �  Q  �  �  �  �  �  �  �  w  l  h  d  `  [  W  S  O  J  E  @  ;  �  �  �  �  �  �  �  �  �  t  S  1    �  �    M  %     �  �  �  �  ]  Y  N  <  1  2      �  �  q  >    �  �  3  S  	o  
  
�  
�  C  w  �  �  �  �  c    
�  
:  	�  �  �  h  �     P  `  [  T  I  :  0  (    
  �  �  �  E  �  z  �  u  �  �  �  �  �  �  z  `  C  +    �  �  �  �  �  �  k  J  &     �  E  <  2  &      �  �  �  �  �  �  �  �  y  e  S  F  b  }    �  8  �  �  �      
  �  l  �  =  q  X    �  #  �  U  ;  �  �  �  �  �  �  f  F  $  �  �  �  H  �  o  �    P  �  �  }  p  e  Z  M  =  -      �  �  �  �  W  7        �   �  �  }  b  P  >  ,    �  �  �  �  �  �  �  �  �  w  O  R  f  m  a  T  G  ;  ,      �  �  �  �  �  u  \  A  %  	   �   �      �  �  �  �  �  x  [  >    �  �  �  �  u  U  *  �  �  �  c  �  �  �  �  �  y  _  ?    �  �  0  �  Z  �  c    �    �  �  �  �  �  �  �  �  t  e  U  ?  "  �  �  b  �      �  �  {  k  a  X  K  <  &  
  �  �  �  v  1  �  �  u  6  �  |  p  d  V  E  4      �  �  �  �  z  \  @  %  
  	      3    �  �  �  �  �  �  x  c  B    �  �  w  ;   �   �   ~   <  �  �  �  �  �  w  O  $  �  �  �  f  0  �  �  �  ^    �  J  '  E  W  Z  Z  \  U  >    �  X  �  3  �  
�  
  	  u  C  �  ;  :  8  8  9  =  B  G  L  O  Q  R  O  L  F  ?  4  )      
)  
+  
  	�  	�  	�  	�  	d  	B  	  �  �  �  �  U  �  �  �    �  �  �  r  S  1    �  �  �  h  4  �  �  j    �    m  �  �