CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�Z�1'      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       Q��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =�"�      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�p��
>   max       @E�z�G�     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vc�
=p�     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @O@           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�L           �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��C�   max       >cS�      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��c   max       B,�F      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�o�   max       B,�$      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�m   max       C���      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�   max       C��z      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          E      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P�CL      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?ࠐ-�      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >�      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @E�z�G�     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vc�
=p�     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @O�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�!-w1��   max       ?ࠐ-�     �  M�                  ,   #               /               .                  	   @         "      m   �         #         -            
   O            ;   �                  (   N4�Oq�ANB�>Q��N��O��O���N4� N���NY�N3OQPX&qO�^�O�8N���O�P;��N�`�N��O�OO�yWOE#�NI�jO�F�N�yO8O�O���O�P���P)��O
��O�rO��qO�gO5i�O�N[�^Nu�NM�N[IO�"O���N	��NP�>PAO�}O<�-O�AM��N�\~N�JEN�fIN9�k���㼃o��o��o��o%   <t�<49X<D��<T��<T��<u<�o<�o<�o<�t�<��
<��
<�9X<�9X<�j<�j<ě�<�`B<�h<�h<��<��=o=o=o=t�=��=�w=49X=<j=<j=@�=H�9=L��=]/=q��=u=u=y�#=�o=��=��=�+=�+=���=��=�"���������������������/2;?BEQJH;/(	"+/#03;<><0(#����BgosoZB5)���#$0<IOMJIC<0&%#[[bt������������tga[B[fosutg[B5)�������������������������

���������ojrt����wtoooooooooo��������������������5BNRLONB5"�M[g�������������ztTM����������������������������������������kdcdnz��������znkkkk1;=PTaz�������zTH?;1T\ajnwz�������znaVT��� 

��������������
#/<KNKC/*#
�����������������������������������������`anz����}zznma``````�||����������������{����������������������
#&((%#
 ���������������������������

�����������(1789;95)���?<<DN[gt�������tg[N?��������������������	
#/<HYYaUOH</#	����������������������������������������������#�������
#/<A@=/&#
�U[chot~���zth[UUUUUU/068BOVWOGB6////////��������������������##/5<=G?</#KFEFIV[ht|�����th[VK��������������������aUOH?<<<DHQUaaaaaaaa������



 ����������������
����������������
"$#
���������"" ���/5AHN^bUNB5.)������������������������������������������������ 	 ��������������

����������������������������N�O�Q�Q�R�N�F�A�:�8�A�K�N�N�N�N�N�N�N�N���z�v�m�T�H�;�/�-�*�+�/�;�?�H�W�a�w�z���нݽ��ݽнʽĽ����ĽȽннннннн�Ƴ������G�E�&����Ǝ�h�����C�[�xƎƳ���������������r�o�f�b�\�f�r�v��������������������������������������������������������������������������������ż��ʼּ�ּѼʼ��������������������������a�h�m�x�w�p�m�f�a�^�Y�V�[�^�a�a�a�a�a�a�����ûͻϻû����������������������������#�/�2�:�3�/�#�"�����#�#�#�#�#�#�#�#����/�H�aÑÔË�z�/���»¥²�����������a�g�i�t�t�p�p�n�f�a�T�C�7�3�6�B�H�L�T�a���!�-�6�:�<�:�-�!���������������������������������������������������������������������������������������������#�(�0�I�R�R�G�@�0���������ľ��������#�n�z�}ÇÉÇÃ�z�n�h�n�n�a�[�[�`�]�[�a�n�5�A�H�N�X�N�A�5�3�4�5�5�5�5�5�5�5�5�5�5�����������������������������x�s�g�]�s���(�4�A�M�T�Z�f�m�f�M�;�(��������(�M�Z�d�b�`�]�Z�X�M�A�=�4�(�!��%�(�4�A�M�<�=�A�?�=�<�/�%�#�"�#�/�/�:�<�<�<�<�<�<àì��������!�"��������ùëèáÝÓà���������������������������������������˾A�M�Z�k�������n�f�Z�M�A�<�4�0�.�4�7�A���������ûλԻ׻׻лû������x�l�W�_�����ʾ׾����� ���׾ʾ������������������#�<ņşşŅ�{�n�U�I��������������������)�B�O�Y�`�c�^�R�@�6�(�������������6�B�I�[�h�t�w�t�_�B�7�:�6�*�,�'�)�&�)�6�������������f�Z�M�L�A�:�3�/�<�M�f�s����M�Z�f�s����{�s�g�f�_�Z�V�M�=�.�)�)�4�Mù������������ùìàÓÏÓÓÚàìîùù��!�-�:�>�H�E�:�6�-�#�!�����������`�m�y�����������y�`�T�G�5�.�(�1�;�E�T�`�@�C�L�P�Y�\�Y�P�L�K�@�:�3�:�@�@�@�@�@�@�-�:�C�F�K�K�F�:�2�-�)�#�-�-�-�-�-�-�-�-�Y�f�o�r�z�������r�f�Y�T�N�Y�Y�Y�Y�Y�Y�������������������������������������'�M�Y�f������}�r�Y�@�4���������������������������������������x�r�o�s������������������������������������������������)�4�)����������������������	��"�T�m�����������m�a�T�H�;�/�"���	D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}DsDuD�D������
��� �������߿ݿѿǿο׿ݿ꿟�����ÿ������������y�m�h�d�h�m�q����������������������������������������������y���������������y�l�h�k�l�r�y�y�y�y�y�y�/�<�H�P�U�a�d�c�a�U�H�<�/�+�$�-�/�/�/�/EuE�E�E�E�E�E�E�E�E�E�EuErEoEoEoEuEuEuEu��������������ؼ޼�������� x E S : 1 > c M d E @ o } W ; 7 * j D I @ 9 x ; v 5 S # D ' | Y ( ; ' $ e Q u q H - r k \  6 T > V 2 H E    �    c  	`  �  9  	  =  �  B  \  �  �  >  �  #  )  *  ;  ~  #  �  �  g  U  �  �  :  �  �  �  �  A  )  }  ;  �  �  t  �  @  �  n  �  �  �  �  D    �    ,  R��C���o�D��>%<#�
=49X=0 �<�C�<�C�<��
<��
=}�<�<��=o<�`B=�+<�/<���=49X=\)=��=o=�j=C�=D��=�o=8Q�>bN>A�7=D��=y�#=��=�7L=�hs=�Q�=ix�=m�h=e`B=q��>1'=��=�o=�7L=�F>cS�=��-=��T=�hs=��P=�j>bN=���B8A��cB%��Bj�B&�B
��B� B!i]B)xB[bB��B�B
�B f�B!}RB`�A���B��B�CB�	BީB �B�Bx�B�B$��B&"B#��Bt{B	�{BIBp�B HB!�'B��B��B�B6B�B#FB�Bv�B��B)�B��B" B@AB3B+'jB,�FB�B�B5[B�A�o�B%�@B?�B&;[B
B�OB!A�B>hB@�B�lB8BB�B >�B!��B-A��B��B�rB��B8�BƋBCkB��B�B$�AB9�B#W�BA`B	��BC�B�BB�B!��BC�BJ9B>BH6B=�B��B�Bg]B?�B@	B��B=fB;LBǚB+>�B,�$B��BCCB@�A��*A��A)&FB"@���A��A�@�P~A���@�L!A�A�=BA�UN@df@�NA�A�#�Aǰ~A���A��BA9�A;�kA�%RA���A�}�A>c@�DAR�A�crA�,uA�:�A@��A>�'A��!@l�Ai@4?�m@z�&@߾�A��@�׶A�Z�A�_A��yA�|>C���A�tAp�A"��A��A�4KC���AǩA���A�sVA)�B�A@��WA��FA�|q@��cA�g@� A�~TA�{�A���@g��@�A��A�l"A�~wA��%A�S@A75bA:�A�Aб`A��A=I@�cAQ!xAꖑA��A؅�A@��A?2A��@k��Aj��?�@|T�@��A҃�@��WA���A��A�gtA���C��A�q�AqJA#)�A��A�uC��zA�9            �      -   #               0               /                  	   A         "      n   �         $         -            
   P            ;   �            	      )               O      #   !               =   !            +         !            %         !      5   )      )                           #            )                                    E                           !                     !                           3                                                #                        N4�NưeNB�>P�CLN�)_O�]�O�?5N4� N���NY�N3OQOlRO�^�N���N��O�O셻N��3N��O���O�yWOE#�N$,	O�o�N�yN�{ O8�O�Pt��O�AkN��nO@�gOH�O�gO�O�˓N[�^Nu�NM�N[IO�>�O���N	��NP�>O��gO.��O<�-O�AM��N�\~N�JEN�fIN9�k  V  �    �  Y  �  V  �  e  l  �  �  O  z  .  �  K  �  �    \  �  '  
	    �  �  �  
$  �    k  �  �    '  �  �  �  c  	�    u  h  	�  L  -  i  <  ;  
  �  1����#�
��o<ě�:�o<D��<#�
<49X<D��<T��<T��=��<�o<�C�<�C�<�t�<��<�9X<�9X<�j<�j<�j<���=8Q�<�h=+=#�
<��=0 �=�j=C�=0 �=<j=�w=H�9=H�9=<j=@�=H�9=L��=�C�=q��=u=u=�O�>�=��=��=�+=�+=���=��=�"���������������������"/9;95/"#03;<><0(#����;Q`ggcNB5"���*)$*0<ILKIG?<0******bbit�������������thb

B[fnstsg[NB5)
�������������������������

���������ojrt����wtoooooooooo��������������������")48BDEB?@<5*)M[g�������������ztTM����������������������������������������kdcdnz��������znkkkkLIHMamz��������zmaTL[Z`alnsz~~zvna[[[[[[��� 

��������������
#/<HMJB/#
�����������������������������������������aanz����zonaaaaaaaaa���������������������{���������������������
"#%&##
������������������������������

����������!+2575)�����MOV[gt��������tg[VPM��������������������!#/1<?GQQMC</)#!����������������������������������������������	��������
#/:=@></#
��U[chot~���zth[UUUUUU/068BOVWOGB6////////��������������������##/5<=G?</#JJMQ[htu~����}xth[PJ��������������������aUOH?<<<DHQUaaaaaaaa������



 �������������������������������

��������"" ���/5AHN^bUNB5.)������������������������������������������������ 	 ��������������

����������������������������N�O�Q�Q�R�N�F�A�:�8�A�K�N�N�N�N�N�N�N�N�T�a�h�m�p�s�m�a�T�H�D�G�H�I�T�T�T�T�T�T�нݽ��ݽнʽĽ����ĽȽннннннн�Ƴ������$�)�����ƚƁ�h�:�3�9�O�mƅƚƳ�f�r�������������w�r�f�d�`�f�f�f�f�f�f������������� ����������������������������������������������������������������Ѽ��ʼּ�ּѼʼ��������������������������a�h�m�x�w�p�m�f�a�^�Y�V�[�^�a�a�a�a�a�a�����ûͻϻû����������������������������#�/�2�:�3�/�#�"�����#�#�#�#�#�#�#�#�/�<�H�U�a�i�n�n�a�U�H�4�#��
���
���/�a�g�i�t�t�p�p�n�f�a�T�C�7�3�6�B�H�L�T�a���!�-�4�9�;�:�-�!��������������������������������������������������������������������������������������������������0�A�A�>�6�#����������������������a�n�z�{ÇÈÇÀ�z�n�f�a�_�^�a�a�a�a�a�a�5�A�H�N�X�N�A�5�3�4�5�5�5�5�5�5�5�5�5�5�����������������������������z�s�g�d�s���(�4�A�M�T�Z�f�m�f�M�;�(��������(�M�Z�d�b�`�]�Z�X�M�A�=�4�(�!��%�(�4�A�M�<�<�@�>�<�<�/�(�$�/�1�;�<�<�<�<�<�<�<�<ù����������������������ùõíóù���������������������������������������˾M�Z�\�f�s�y�|�s�f�d�Z�M�B�A�4�4�4�A�E�M���������ûȻлһл˻û������������z�����ʾ׾����� ���׾ʾ����������������<�U�{ŒŔň�{�n�U�0������������������<��6�B�H�Q�R�N�G�B�6�)��������������6�=�B�E�[�`�h�o�h�[�O�B�9�;�6�)�'�)�/�6�M�Z�f�s�����������s�f�Z�M�A�;�:�A�I�M�Z�f�s�{�~�x�s�f�a�Z�M�D�5�4�3�4�6�A�K�Zù������������ùìàÓÏÓÓÚàìîùù��!�-�:�:�D�@�:�.�-�!������������m�y�}�����������y�m�`�T�G�8�/�:�H�T�`�m�@�C�L�P�Y�\�Y�P�L�K�@�:�3�:�@�@�@�@�@�@�-�:�C�F�K�K�F�:�2�-�)�#�-�-�-�-�-�-�-�-�Y�f�o�r�z�������r�f�Y�T�N�Y�Y�Y�Y�Y�Y������������������������������������'�@�Y�f�w�z�s�r�Y�M�@�4�������� ��'�������������������������������x�r�o�s������������������������������������������������)�4�)����������������������;�T�a�m�~�������������z�a�T�H�;�/�#�/�;D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������
��� �������߿ݿѿǿο׿ݿ꿟�����ÿ������������y�m�h�d�h�m�q����������������������������������������������y���������������y�l�h�k�l�r�y�y�y�y�y�y�/�<�H�P�U�a�d�c�a�U�H�<�/�+�$�-�/�/�/�/EuE�E�E�E�E�E�E�E�E�E�EuErEoEoEoEuEuEuEu��������������ؼ޼�������� x , S 3 1 @ c M d E @ P } R @ 7 ' 6 D H @ 9 ~ 6 v , D # B  x 0 1 ; & % e Q u q A - r k ;  6 T > V 2 H E    �  �  c  �  �  H  �  =  �  B  \    �  '  �  #    �  ;  r  #  �  �  a  U  
  �  :    O  5  �  �  )      �  �  t  �  �  �  n  �  �  l  �  D    �    ,  R  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  V  T  R  O  M  K  I  G  D  B  B  D  F  H  J  L  N  P  R  T  f  p  l  a  V  W  b  �  �  �  v  Q     �  �  g  �  �  P               �  �  �  �  �  �  �  �  �  �  z  P  '   �   �  �  V  �  �  �  �  x  L    �  �  a    #  Y  H  �  �  g  �  T  V  X  Y  X  W  T  P  G  B  E  >  2      �  �  �  �  k  W  �  �  �  �  �  �  �  �  �  c  4  �  �  Z  �  �    S  `  A  Q  D  =  8  &  �  �  �  �  �  G  �  �  u  :  �  �  0  (  �  �  �  �  �  �  �  �  �  �  ~  n  _  M  7  !    �  �  u  e  ]  V  N  F  :  /  #      �  �  �  �  x  U  N  M  M  L  l  p  t  v  u  t  o  h  `  Y  R  K  F  A  �  `  �    i   �  �  �  �  �  �  �  �  �  �  ~  _  @  "    �  �  �  �  d  C  �       
      �  �  l  �  �  ~  W    �  �  _  �  p  �  O  B  4  $    �  �  �  �  h  ;    �  �  �  �  U    �  �  u  y  y  v  q  j  f  a  [  Q  G  =  .    �  =  �  m    �  -  .  *       �  �  �  �  �  ^  2    �  �  R  �  �     �  �  �  �  �  �  �  �  �  _  ;    �  �  �  �  i  K  .    �  �    (  <  I  J  @  *    �  �  �  �  �  �  �  c  �  �   �  >  9  4  ;  ]    �  �    p  _  N  :  '    �  �  �  y  H  �  �  �  �  �  �  �  �  �    z  v  q  k  b  Y  Q  H  ?  6  �    �  �  �  �  �  �  �  �  �  q  Z  9    �  �  k  I  �  \  G  1    �  �  �  �  }  `  C  %    �  �  �  �  �  b  .  �  �  �  �  �  �  �  �  u  j  e  N  5      �  �  R     �      #  '  (  %        �  �  �  a  E  )    �  �  �  �  �  	  	�  	�  	�  
	  	�  	�  	�  	�  	L  �  �    �     N  n  l  �      �  �  �  �  �    �  �  �  �  �  �  �  y  b  K  3    j  z  �  �  �  �  �  �  t  _  J  1    �  �  �  x  0  �  �  �  �  �  �  �  �  �  �  �  �  �  Q    �  �  C  �  |    �  �  �  �  �  �  �  s  ^  G  -    �  �  �  �  s  W  <  >  q  	�  
  
#  
  	�  	�  	�  	�  	�  	q  	I  	&  �  �  a  �  �  �  E  �  �  �  @  �  /  �  �  �  �  �  <  �  K  �    6  
�  �  �    �          �  �  �  {  `  p  �  �  �  �  �      .  A  �    3  Q  b  j  k  h  `  S  9    �  �  y    �    �  �  �  �  �  �  �  �  �  �  �  �  k  /  �  s     z  �  [  �   �  �  �  �  �  �  �  �  �  �  d  8  �  �  Y    g  V  �  '  �  �  �  �        �  �  �  �  ^  !  �  �  V    �  ;  �  �    "  $      �  �  �  �  ^    �  x    �     �  �  %  �  �  z  W  2    �  �  �  s  G    �  �  �  �  Q     �  X    �  �  �  y  m  `  O  A  Z  _  B     �  �  �  >  �  �  @  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  c  Z  R  D  5  '    
  �  �  �  �  �  �  |  ^  9    �  �  	:  	v  	�  	�  	�  	�  	�  	a  	3  	  �  �  K  �  �    q  v  #  a      �  �  �  �  �  �  |  g  P  7    �  �  �  �  y  :   �  u  z  �  �  �  �  u  a  M  9  "    �  �  �  �  }  `  B  $  h  X  I  6      �  �  �  �  b  =    �  �  �  v  K      �  	[  	u  	�  	�  	�  	�  	P  	  �  }  %  �  c  �  x  �  M  �  �    b  �  6  �  �  �  #  A  L  1  �  �    X  �  R  �  �  
u  =  -      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  2  �  i  R  ;  #    �  �  �  �  e  @    �  �  �  x  A  �  j    <  8  4  /  +  &  !        
    �  �  �  �  �  �  �  �  ;  1  '      �  �  �  �  �  �  r  [  D  .  �  y      �   �  
  �  �  �  t  N  )    �  �  �  `  ;    �  �  �  �  C  �  �  �  c  @  .    �  �  C  
�  
j  	�  	l  �  l  �  Y  z    �  1  )      �  �  �  v  =    �  �  N    �  �  @  �  �  Z