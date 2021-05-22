CDF       
      obs    2   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?tz�G�{   max       ?�-V      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�H�   max       P���      �  t   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �D��   max       >
=q      �  <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @D�z�G�     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��p��
>    max       @vZz�G�     �  '�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @O@           d  /�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�$@          �  0   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >�p�      �  0�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�W6   max       B%�-      �  1�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B%�:      �  2`   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >$�I   max       C���      �  3(   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?�   max       C��      �  3�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         v      �  4�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          %      �  6H   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�H�   max       P�|      �  7   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�/��v�   max       ?�W>�6z      �  7�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �D��   max       >\)      �  8�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @D�z�G�     �  9h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���\*    max       @vZz�G�     �  A8   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @O@           d  I   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�^�          �  Il   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  J4   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�n��O�<   max       ?�P��{��     �  J�         
  v   I               %                           P   %   
      $      K               �                  E         ,            	      
      	         	N��tNN��O
�'P���P�=�Oj�3Nh�OGV�N��P$;�N���N��N��NC �N�jO^рO���O\
�Po�AO��N���O!$�O���O8�<OhC�Nj�EO�)�N�jmO�Pb��N]�^O�E�O��vN���O{fO�ЧM�H�NGǱO���N���O�(O(�N���N��SN=ҍN�YNB��NfS�N"N)���D���49X�o�D��;o;D��;�`B;�`B<t�<49X<D��<u<u<�o<�C�<�t�<�9X<ě�<���<���<���<��=+=\)=t�=�w=,1=,1=,1=0 �=D��=D��=P�`=Y�=]/=]/=]/=aG�=u=}�=�%=�%=�%=�+=�O�=�t�=�9X=�^5>$�>
=q��������������������!"#$0<=A?<0#!!!!!!!!������	����#"*B[����������gNB1#\Zaz�������������me\�z|�������������������������������������gnuz������������zungZUV[dht}~vthhg[ZZZZ����
#/;FPQNF/#����������������������Y[gpt|�toge][YYYYYY��������������������739<HJTRH<7777777777�����������������������������������������������������������������������������������
)BNZ_\[NB5) ��403;O[hpuywtkh[WOIB4NOVY[chkproh[WPONNNN���������������������)5=EJJEB)
��	"/;>FE;/&"	��������

��������������������������������������������������

�������
#),*$#
����� :LSQThrn`O4)��#/<=<55/-#������)3.)!���(;HOau��������zaU;9(JFN[gjnkg[VNJJJJJJJJ��������������������������������������������������������������������������������adjt{������������zma[UTVX[^hnttttjh[[[[[;BLTYaedc`[WTKHC;69;��������������������������	�������������������������������������������������nuz��������zvpnnmnnn"#/<HHHH</%#"""""""""#''#��������������������--/<EH<<5/----------ŇŔŠŦŭŰŭŧŠŔŎŇńłŇŇŇŇŇŇ�����������������������������������������!�-�3�E�F�S�Y�S�F�:�-�!������
��!��6�O�j�u�r�n�[�B�6�)�����������������������������������Z�G�5�(�����[�g���tāčĚĦĭįĪĦĚčā�r�h�b�]�h�l�p�t����������������ùùùû�������������������(�.�A�T�_�]�Z�N�A�5�&��������������������¿����������������������������(�A�g�y�{��{�g�Z�5���������àíùÿ��ýùìàÓÇÃ�z�u�zÇÓÖÛà���������������y�n�m�k�m�y���������������b�o�{ǈǐǈǇǀ�{�o�g�b�\�a�b�b�b�b�b�b������������������������������������������
������ܹʹù��ùϹܹݹ���(�4�A�M�Z�n�s���~�f�Z�M�4�(������'�4�@�M�U�]�f�l�w�4������������'�����������¿ǿſ��������������|��������Ƨƹ�������������Ƨ�u�[�M�K�P�\ƁƜƧ�������ʾܾ��׾ʾ������f�R�B�M�Z�f�q�����(�4�?�4�1�(���������������������������������������������������~��������������������¿²¦¦²¿���T�a�m�z���������|�z�m�a�[�V�T�T�O�R�T�TDoD{D�D�D�D�D�D�D�D�D�D�D�D{DqDeDbDbDhDo�.�;�G�O�P�G�;�.�"��"�&�.�.�.�.�.�.�.�.�H�a�n�zÇÜÚÇ�n�a�H�>�/�#����#�)�H��"�.�;�E�;�1�.�"����	�������	����m�y�����������y�m�`�T�G�E�A�G�L�T�`�d�m���4�L�Y�f�r�r�f�Y�>�4����ܻû����ܼ�g�l�q�q�n�g�d�[�N�L�N�W�[�d�g�g�g�g�g�g�L�Y�r�~�������������������r�e�L�9�:�E�L�	���"�'�"����	������������������	����������������������������������������������������&��������������������ҹϹܹ���� ����ܹ��������������������ϾA�M�S�Z�\�Z�M�A�@�?�A�A�A�A�A�A�A�A�A�A�6�C�O�\�`�e�\�O�C�6�4�3�6�6�6�6�6�6�6�6���������)�.�.�)�������������������޼��������ʼͼʼƼ������������������������n�{ŇŒŔřŔŇ�{�n�b�U�O�I�G�I�L�U�b�n�����(�4�;�@�?�5�4�(�'��������"�/�;�?�F�;�5�/�"������"�"�"�"�"�"�����������ĽĽĽ������������|�z�����������
�
�����
�����������������������������������ܻٻ׻ܻ��������������������������������������������*�6�C�O�Z�O�O�C�6�-�*�&�*�*�*�*�*�*�*�*E7ECEPETE\EcE\EPECEBE7E4E7E7E7E7E7E7E7E7Ŀ��������ĿĳĳĳĳĿĿĿĿĿĿĿĿĿĿ C - N  C  K ` / * C 3 / 0 D X W 0 > H ? ` Y B - ; } c 9 2 Q K L 5 h M L d 4 6 b : I c a H X ; i /  �  l  H  r  F  �  x    �  �     �  �  P  -  �  �  �  #  2  �  �  �  �  �  �  �  �  R  �  �  �    �  �  �    �  �  �  �  a  �  �  z    �  |  e  7���
��`B:�o>�p�=���=+<�t�<���<�1=@�<�`B<�j<���<�/<�/=8Q�=Y�='�=�
==}�=C�=49X=�C�=e`B=�G�=<j=y�#=@�=e`B>7K�=]/=���=��=m�h=��-=��m=q��=y�#=���=���=���=��-=�t�=���=���=�-=ě�=��>O�>n�BWB%�-B�(B	UkB �B9B�BMVB��B��B!�5B	uB��BB �B"��B!WPB��B>ABtB~gB>�B�_A�W6B��B�B[B��B�Bb�B��B�B�5B�B)sBF�B�B~�B ��B��A��B0RBF�B�aB�	Bk�Bv�BY�B�WB�KBzB%�:B�^B	BB ��B@�B?B��B��B��B!�B	��B�^B?�B��B"�8B!B�B�tB��BA�B�AB��B�vA���B�0B4�B?�BI=B?�B3|B;�B��B)PB�BF�B?�B��BV�B lB�A�eiB5�B>�B D�B>�BPuB�B:�B?B��A��@���@s9Aձ�A�%�A�e�Aϒ"A�knAs�hA���A�q�An1�B�A�UG?!�5A;y�@�c+As�B��AJ��A4��A�޳A�rA�;KC��GAb��AŏA^fgAj�@���A�2�?��`A�� A���A��>$�IA<��B�A�S�@��\A�8A6
A�(A!C�A�T9@��tA.g�B �RC���A��
A�@��@ssA�r�A�Y�Aނ�Aς3A�uAs0uA��xA�T�AmB��A��?A:�A;��@�-�Au}B �AJ�$A3�A���A���A��jC��^Ac�A�x~A_"�Ai	@���A�}�?�!HA�+�A��A��>?�A=SB=�Aӕ�@���A�n�A5#"A�~�A!��A��b@�z�A-�,B �mC��A��r         
  v   J               %      	                      Q   &   
      $      K               �                  F         -            
            	         	            5   5               )                     #      3   %                     !         5         #         !                                                      %   %                                          %                        !         !         #                                                   N��tNN��N�)P�|O�55OQ�NI!�O:�Nw�=O��8N���N��N��NC �N�:O^рO���O٣P�)O~��N���N��POK9�OץO l4Nj�EO�)�N�jmO�O��wN]�^O�E�O��vN���N��gO�f�M�H�NGǱOm.N���O�(O(�N���N��SN=ҍN�YNB��NfS�N"N)��  %  :  4  �  n  ]    �  �  @  c  �  �  %  �  h  c  �  �  �  0  �  �  &  d  t     �  �  �  �  ?  �  �  &  �  �  ^  �    �  �    �  E  B  Q  �  R  üD���49X��`B>\)<��<D��<o<o<D��<���<D��<u<u<�o<�t�<�t�<���<�`B=L��=t�<���=C�='�=�P=Y�=�w=,1=,1=,1=�v�=D��=D��=P�`=Y�=e`B=��=]/=aG�=�hs=}�=�%=�%=�%=�+=�O�=�t�=�9X=�^5>$�>
=q��������������������!"#$0<=A?<0#!!!!!!!!���������?88@N[gt�������tg[N?lmrz�������������uml����������������������������������������wz�������������zxnwYZ[hqtvzztph_[YYYYYY���#/59@>;4/#
����������������������Y[gpt|�toge][YYYYYY��������������������739<HJTRH<7777777777��������������������������������������������������������������������������������5BHNSSRKB5)@;::>GO[hkoqqqnh[OE@NOVY[chkproh[WPONNNN��������������������
	)59AFFCB5)	 	"/9;CB;/" 			����������

�������������������������������������������������

�������
#),*$#
��� �)6BOT\ZQB6 #/<=<55/-#������)3.)!���(;HOau��������zaU;9(JFN[gjnkg[VNJJJJJJJJ��������������������������������������������������������������������������������yvxz|�������������zy[UTVX[^hnttttjh[[[[[;BLTYaedc`[WTKHC;69;��������������������������	�������������������������������������������������nuz��������zvpnnmnnn"#/<HHHH</%#"""""""""#''#��������������������--/<EH<<5/----------ŇŔŠŦŭŰŭŧŠŔŎŇńłŇŇŇŇŇŇ�����������������������������������������-�1�:�D�F�S�V�S�F�:�-�!�������!�-���)�B�S�Z�]�[�V�K�6����������������������������������s�Z�A�1�*�+�4�N�g����čĎĚĦħĨĦģĚčā�t�p�o�t�wāĊčč����������������ûÿ���������������������(�,�5�A�N�S�^�\�Z�N�A�5�'�������(�����������������������������������������(�5�>�N�[�Z�U�N�A�5���������� ���(àíùÿ��ýùìàÓÇÃ�z�u�zÇÓÖÛà���������������y�n�m�k�m�y���������������b�o�{ǈǐǈǇǀ�{�o�g�b�\�a�b�b�b�b�b�b��������������������������������������� ���	�������޹ܹϹ˹Ϲܹ����(�4�A�M�Z�n�s���~�f�Z�M�4�(������4�@�M�P�Y�a�e�M�@�4����������)�4�����������ÿ���������������������������ƎƧƳ����������������ƳƎ�u�e�`�c�uƁƎ�����������ʾҾ׾Ҿʾ����������v�j�t������(�4�?�4�1�(������������������������������������������������������²¿��������������¿²¦¦¯²�T�a�m�z�~�����~�z�x�m�a�]�X�U�T�P�T�T�TDoD{D�D�D�D�D�D�D�D�D�D�D�D{DyDoDkDjDkDo�.�;�G�O�P�G�;�.�"��"�&�.�.�.�.�.�.�.�.�H�a�n�zÇÜÚÇ�n�a�H�>�/�#����#�)�H��"�.�;�E�;�1�.�"����	�������	����m�y�����������y�m�`�T�G�E�A�G�L�T�`�d�m����'�7�C�I�I�C�:�4�'���������g�l�q�q�n�g�d�[�N�L�N�W�[�d�g�g�g�g�g�g�L�Y�r�~�������������������r�e�L�9�:�E�L�	���"�'�"����	������������������	�������������������������������������������������������������������������ҹùϹܹ������ܹϹù������������������þA�M�S�Z�\�Z�M�A�@�?�A�A�A�A�A�A�A�A�A�A�6�C�O�\�`�e�\�O�C�6�4�3�6�6�6�6�6�6�6�6������	���!�(�)�"����������������뼘�������ʼͼʼƼ������������������������n�{ŇŒŔřŔŇ�{�n�b�U�O�I�G�I�L�U�b�n�����(�4�;�@�?�5�4�(�'��������"�/�;�?�F�;�5�/�"������"�"�"�"�"�"�����������ĽĽĽ������������|�z�����������
�
�����
�����������������������������������ܻٻ׻ܻ��������������������������������������������*�6�C�O�Z�O�O�C�6�-�*�&�*�*�*�*�*�*�*�*E7ECEPETE\EcE\EPECEBE7E4E7E7E7E7E7E7E7E7Ŀ��������ĿĳĳĳĳĿĿĿĿĿĿĿĿĿĿ C - F  M  B _ (  C 3 / 0 < X O $ > 7 ? L ? G $ ; } c 9 # Q K L 5 b L L d  6 b : I c a H X ; i /  �  l  !  \  n    X  �  �  ?     �  �  P  �  �  2  >  �  �  �    �  h  X  �  �  �  R  �  �  �    �  8  -    �  �  �  �  a  �  �  z    �  |  e  7  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  %        �  �  �  �  �  �  x  \  8    �  �  �  ~  X  2  :  7  4  2  /  +  &             �  �  �  �  �  �  �  �       2  /  )  &  $        �  �  �  �  �  �  ~  j  p  w  �  �     �  �  �  �  �  �  �  0  �  �  5  �  p  �  �  �  ^    �  �    6  P  `  k  k  Y  9  
  �  �  .  �  1  n  0  $  m  �  	  .  J  X  Z  P  :    �  �  �  c     �  R  �  �  �  �  �      	        �  �  �  �  y  B    �  �  y  E    �  �  �  �  w  a  O  3    �  �  �    R    �  f  �  U   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  >    �  �  c  �  �       ,  5  :  >  >  8  ,      �  �  �  G  �  6  V  c  [  R  C  1      �  �  �  f    �  S  �  ~    �  t  �  �  �  �  �  �  �  �  �  �  �  �  �  {  q  g  ]  T  M  M  L  �  �  �  �  �  ~  p  `  N  <  (    �  �  �  �  O    �  �  %  (  *  )  %      
  �  �  �  �  c  7    �  �  H  �  �  �  �  �  �  n  U  ;       �  �  �  �  �  �  �  �  �  �  �  h  Z  F  /      �  �  �  �  �    \  ,  �  �    �  3  �  N  V  `  `  Z  O  :  +    	  �  �  �  �  e  8    �  �  u  A  b  |  �  �  �  �  x  i  Y  H  3    �  �  �  T    �  [  �  %  n  �  �  �  �  �  �  �  �  w  6  �  l  �  D  n  b  �  �  �  �  �  �  �  �  �  �  �  �    Q    �  x  	  �    C  0  $    
  �  �  �  �  �  �  �  m  V  ?  ,        �  �  �  �  �  �  �  �  �  �  �  r  S  2    �  �  �    Z  &  �  S  �  �  �  �  �  �  �  �  q  0  �  �  X    �  R  �  Z  �  %    #        	  �  �  �  �  b  9    �  �  �  x  5  �  U  7  >  U  a  `  L  &  �  �    �    s  �  /  
w  	�  �  ?  t  r  o  k  b  Z  P  E  ;  .  "    
  �  �  �  �      *     �  �  �  �    �  �  c  @  %  �  �  �  /  �  o  �  x   �  �    n  \  J  3      �  �  �  �  �  u  L  "     �   �   �  �  �  �  �  �  �  �  k  T  6    �  �  �    T  *    �  �  	&  
@  
�  !  k  �  �  �  �  �  T  @  ;  &  
�  
}  	�  �  �  !  �  �  �  �  |  o  a  N  8  !  
  �  �  �  �  �  �  o  Z  F  ?  (    �  �      �  �  �  �  _  '  �  �  M  �  �  7  �  �  �  q  l  z  w  p  g  Y  H  0    �  �  �  ]  B  %      �    w  p  h  ^  U  K  I  Q  Z  b  c  `  \  X  9    �  �  !  #      �  �  �  z  C  
  �  �  I  �  �  Q  �  �  ;  �  "  _  �  �  �  �  Z    
�  
o  
  	�  	F  �  Q  �  �  �  E  v  �  �  �  �  �  ~  m  ]  N  A  4  (  !                ^  e  m  t  s  q  o  j  c  ]  S  G  ;  /  $    
  �  �  �  Y  v  �  �  �  �  �  �  �  Q    �  �  C  �  m  �  j  �  �      �  �  �  �  �  �  �  k  R  3    �  �  �  �  �  �  �  �  �  �  �  �  �  Q  '    �  �  �  �  �  f  J  '  �  �  @  �  �  �  �  �  �  �  �  �  y  c  K  0    �  �  �  �  n  0    �  �  �  �  �  �  �  �  �  w  Z  :    �  �  �  n  @    �  �  �  i  R  :  "  
  �  �  �  �  �  �  �  d  E  %    �  E  +    �  �  �  �  d  5  �  �  �  Q    �  �  =  �  �  \  B    �  �  �  �  �  m  R  6    �  �  �  �  k  =    �  1  Q  K  D  =  5  -  $        �  �  �  �  �  �  �  �  �  �  �  q  Q  0    �  �  z  L  -    �  �  �  �  r  Q  /    �  R  9  !    �  �  �    R  @  4  %    �  �  �  �  o  O  .  �  �  �  r  d  V  G  8  (      �  �  �  �  o  #  �  �  6