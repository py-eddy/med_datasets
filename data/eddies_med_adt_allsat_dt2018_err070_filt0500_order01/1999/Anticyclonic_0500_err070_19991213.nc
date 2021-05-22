CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�~��"��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�J#   max       Q�f      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �49X   max       =Ƨ�      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���R   max       @F~�Q�     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @v[��Q�     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @K�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @�O           �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���
   max       >�Z      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B2�J      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��5   max       B3o�      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�g   max       C���      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�4b   max       C��U      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         !      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          S      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          U      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�J#   max       Q
�      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�X�e+�   max       ?垃�%��      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �49X   max       >��      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��G�{   max       @F~�Q�     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v[��Q�     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @K�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @�n           �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         GC   max         GC      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�n��O�<   max       ?垃�%��     �  Pl            
   !            3         3               ]         '   @         
      U   
   R         E     !            ,         4   3   _         (               ]   >               *N��0N�.&N��+N+�0OH�`O5��On�=M�J#O��O�8rN���O�l9N��N�SoO�^O�Q�fO9ƠN	�O�?O��1NT\�NR0UN�>�N�(P�olNR��O۞UO50~N��lP��M��`Pe*�M���N-�N�P%�9O4`�N��O�P]O�6kPk�eO���Nrl*O��O��O[N��O�PN<�O���N�;O��N�7O�O	!��49X�o�o%   ;o;D��;D��;��
;��
;ě�;�`B<o<#�
<49X<49X<D��<u<�C�<�t�<�t�<���<��
<��
<�1<���<�`B<�`B<�`B<�`B<�h<�<�=+=\)=t�=�P=�P=�P=�P=��=�w='�='�=,1=8Q�=P�`=Y�=]/=e`B=m�h=�%=�%=�o=�o=�O�=Ƨ����������������VX[gty�����ttge[VVVVjkrt�������tjjjjjjjj;768<GHQIH?<;;;;;;;;#/3<HPMMMG</#��)5A>50)������������������NNFKO[[][NNNNNNNNNNN/<HUanzn\UH</'���)5BLPQQNB5���������������������������������556BBOS[ba[ZOLB65555,//<HUU]UHH<7/,,,,,,jfnz�����������zumajGEKOThtz�����~thXROG����K[uwgbN5����������!)25;75)���tlot����uttttttttttt���
#<ITZXP=,��(.6J[ht�����th[O80,(xnz������zxxxxxxxxxx�������������������������������������������������������������������
#9</#
�����������������������������������������������������������������#*-56B>:6*)��g�����{ytgaN96(���������������������#)5[gt�����tNB5++/2<HHQH</++++++++++{��������*+*)")**********vqosz�������������zv��������������������������������YUX[]blz���������zaYz�����������������zz�������5R_ebN�����������������������chuu�������ujhcccccc���*6OT[^YO6)�����������������������������������������6689@BKOUUTROGB66666ssty�������������wts�������%*&����������
/<EHIHE</#
�#&'##����������������������������������������nz~����������znkedfn������� �������ڹ���������	��������޹�����������I�U�V�X�^�U�J�I�<�0�-�0�0�8�<�H�I�I�I�I�L�Y�e�j�q�f�e�Y�T�L�D�C�L�L�L�L�L�L�L�LECEPE\EiElEiEhE\EPEOECE;ECECECECECECECECD�D�EEEEEEE&E,E*EED�D�D�D�D�D�D�`�i�k�f�g�j�k�`�T�K�G�B�;�:�4�8�;�G�R�`�A�N�Z�e�s���������������s�Z�H�A�8�6�7�A�l�l�y���������y�w�l�l�l�l�l�l�l�l�l�l�l����&�#���
����������������������<�U�a�d�^�a�c�_�U�H�<�/�!����
��/�<�)�+�*�2�)�������&�)�)�)�)�)�)�)�)�S�_�l�x�������������l�_�S�:�-�$�*�:�F�S���ʼʼּ׼����ּ̼ʼ����������������	����!���	������������	�	�	�	�	�	���������������������������������������׾A�M�Z�a�f�q�j�f�Z�M�A�4���	�	��(�1�AƧ���!���ƱƎ�u�O�6�#�"���������C�uƧ�f�s�x�v�w�u�s�p�f�b�Z�M�K�E�B�I�M�[�b�f�����������������������������������������4�@�M�f�r�����������������f�M�4�$�'�4����4�A�F�E�4�'���ܻλǻŻǻлܻ������������������������������������������������¾¾������������������������������ûĻ˻лڻֻлƻû������������������û��{ǈǔǡǦǭǡǘǔǈ�{�o�n�i�o�p�{�{�{�{����5�v¡�g�B�)�������������������������¿²§²´¿�����������������˹Ϲܹ������#�#�����ܹù�����������Óàìùÿ����þùàÓÇÅ�z�x�y�z�ÇÓ�`�l�u�y�������y�x�l�`�S�R�S�S�\�`�`�`�`���5�A�z���s�Z�(��ѿ������������ѿ���C�O�\�]�a�\�O�G�C�B�C�C�C�C�C�C�C�C�C�C����)�3�;�E�C�8�)����������ôéîû����²»¿��¿¿²±­®²²²²²²²²²²�/�<�D�H�K�H�<�/�*�/�/�/�/�/�/�/�/�/�/�/�?�3�'�&�!�'�3�=�@�B�?�?�?�?�?�?�?�?�?�?�Z�����������������������s�Z�0�(�(�7�I�Z�������	�����	���������������������3�@�F�L�N�O�L�E�@�3�+�,�/�2�3�3�3�3�3�3ŇŔŠŹ����������������ŹŭŠŔŋŃ�}Ň�g�����������|�f�Y�M�@�6�3�2�4�@�M�b�g��������������	��������������|�z�������"�.�<�T�W�V�X�W�T�G�;�"��������"����	�	�	�	��������������������������������������������z�m�_�V�X�a�m�x���������Ľнؽ�ݽнĽ��������y�l�n�w�����h�s�s�j�g�Z�M�7�(����%�+�4�A�M�Z�d�h���������ɺκֺںֺɺ��������������������!�-�3�:�>�=�8�-�&�!����������!Ŀ�����
��<�P�Y�W�I�0��������ĳĦěĞĿ�`�m�y�������������y�m�`�S�G�:�/�.�G�L�`���	���!��	��������������������������������	���"�%�"���	�����������������/�;�H�L�T�Z�X�T�H�<�;�2�/�,�/�/�/�/�/�/����
����������޻ܻٻֻܻ����DVDbDoD{D~D�D�D{DoDbDVDIDEDFDIDIDVDVDVDV L X 4 V ; V ^ : U 0 4 ; j C G . H l 9 X * ` V Z N 3 n @  9 O > # m 6 A L g V 0 . > 6 g G & _ d $ U 0 E M Z H   �  �  �  \  �  �      �  �  �  �  �  �  �     	a  �  (  ~  A  z  �  �  �  V  ~    �  �  i  %  �  ]  /  0  -  �  �  |  k  "    �  �  �    �  2  �  �  <  T  �  O  +���
;o<�t�<t�=C�<#�
<ě�<#�
=e`B=+<T��=q��<�t�<�C�<�h=o=�/<��<�1=e`B=���<���<���<��=#�
=�`B=�P=�;d=ix�=+=ȴ9=C�>�Z=#�
=#�
=,1=��
=Y�=0 �=�E�=�Q�>
=q=�%=D��=�{=��
=�%=y�#=�O�>��>   =�C�=��w=�\)=Ƨ�>O�B��B	�`B��B�rB%BSB��B�B� B�KBf�B��B��B�IBBB��BK�B�B|B%�4B%�B�BAB!��B=�BB�uB�B!�KB/֔BxRBB��B�B$TB��B'B�@BjA���B��BɻB!�9B2�JBnB+ͣB��BK�B=)B�B��B|�B�B�}B�FB��BE�B	��B��BʭB�B��B�\B�|B��B�6BS�B��BA�B�UB�-B��B�_BBB�vB%��B�B�LBA?B" bBA�BEtB,�B�pB"D�B/�qB��BB��B�B={B��B_B��B�`A��5B.�B�gB!��B3o�BŔB+�fBȉB:�B�ZB>�B��B��BH;B�{B@B��?8"A�$2?֫jC���C�K�Ag��A���AI�A���AèA�q�@�.:@�"�A�ԴA���A;��B��A@hA���@��@���A�zAM�@�1�B�`A��A��X>�gA�FYA��A��BK�Aһ�A�x�A�NL?�U'A���A���?�2
A�=8@ڢ7A��Aa;AXiA�(DA!��A;��@*�$@lx�A���Aj_�A[pTA��A��R@�d�C���?(n:A��?�
C��UC�H2Aeo�A���A�AҎ�A���A��?@�+@��MA���A�nA=�BFA?
sA���@�Hf@�TA��AL�@��B��A�[�A�-�>�4bA�L*AwOA�ylBI=AҘ?A�~A�U�?�f�A���A�=�?�E�A���@��A��EAaxAYz�A���A�bA:��@'��@e��A�m�Ak�A[GA�O0A�$�@� C���            
   !            4         4   	            ^         '   A               U   
   R         E     !            ,         4   4   _         )               ]   ?               *                        
      #      %         #      S         )   '               9      #         ?      /            +            !   1            #            3                                          
      !               #      U                           +               /                  '               /            !            !                  N��0N�.&NMv^N+�0N�M�O5��On�=M�J#N�M�O��~N���O��N��N�SoO�^O�NQ
�N��N	�O^lO��FNT\�NR0UN5��N�(P3�QNR��O�V�O&�-N��lP7��M��`O���M���N-�N�Pu�O4`�N��O;~rO(0PS��O���Nrl*Ob3�O�9OO[N��O�O��GO~��N�;O��N�7O:�O	!�  �  _  �  �  �  �  \  
  �       h  �  �  `      �  7  �  �  �  t  �  �  1  j  8    �  �  r  d  I  �  K  �      �  0  	�  7  �  �  �  �  !  �  0  	�  n  +  f  �  ׼49X�o:�o%   <e`B;D��;D��;��
<�`B<t�;�`B<�<#�
<49X<49X<���<�C�<�1<�t�=�w<�<��
<��
<ě�<���=D��<�`B=T��<�<�h=D��<�>��=\)=t�=�P=#�
=�P=�P=aG�=y�#=@�='�=,1=e`B=aG�=Y�=]/=e`B=�-=��-=�%=�o=�o=�\)=Ƨ����������������VX[gty�����ttge[VVVVnltt�������tnnnnnnnn;768<GHQIH?<;;;;;;;;#/<@B=</#��)5A>50)������������������NNFKO[[][NNNNNNNNNNN*&$&/<EHPUPH</******� 5BIMONK=5
�����������������������������	
����556BBOS[ba[ZOLB65555,//<HUU]UHH<7/,,,,,,jfnz�����������zumajVSOS[ghitz~{yth^[VV���HW[sve`N)����������)-575/) ��tlot����uttttttttttt#%0<EGFA80+#2//5O[ht}}yyzvh[OB62xnz������zxxxxxxxxxx�������������������������������������������������������������������
#/1&
�����������������������������������������������������������������#*-56B>:6*)5[g{�ztng[SNB5)��������������������2/18BN[gt����tg[NB82+/2<HHQH</++++++++++{��������*+*)")**********xusrvz������������zx��������������������������������feeflmz����������zmf��������������������������BN\a^NB�����������������������chuu�������ujhcccccc)6BOTWROB6)����������������������������������������6689@BKOUUTROGB66666ssty�������������wts�������!"�����
#/5<@CEB</#��#&'##����������������������������������������kfdfnqz}���������znk������� �������ڹ���������	��������޹�����������I�U�V�X�^�U�J�I�<�0�-�0�0�8�<�H�I�I�I�I�L�Y�e�e�n�e�\�Y�X�L�G�G�L�L�L�L�L�L�L�LECEPE\EiElEiEhE\EPEOECE;ECECECECECECECECD�D�EEEEED�D�D�D�D�D�D�D�D�D�D�D�D�`�i�k�f�g�j�k�`�T�K�G�B�;�:�4�8�;�G�R�`�A�N�Z�e�s���������������s�Z�H�A�8�6�7�A�l�l�y���������y�w�l�l�l�l�l�l�l�l�l�l�l����������������������������������U�a�b�\�`�[�U�H�<�,�#������#�/�<�U�)�+�*�2�)�������&�)�)�)�)�)�)�)�)�S�_�c�l�x�}���x�l�l�_�S�F�B�:�5�8�:�F�S���ʼʼּ׼����ּ̼ʼ����������������	����!���	������������	�	�	�	�	�	���������������������������������������׾4�A�M�Z�c�f�g�f�a�Z�M�A�4�(� �!�(�*�4�4Ƨ�������ƪƎ�u�O�6�%�#���������C�uƧ�f�s�s�s�s�s�o�k�f�\�Z�V�M�J�I�M�M�Z�a�f�����������������������������������������M�Y�f�j�r�������������r�f�Y�O�M�H�I�M������2�9�>�;�4�'����ܻλ˻ͻлܻ������������������������������������������������¾¾������������������������������ûƻϻлѻлû����������ûûûûûûû��{ǈǔǡǦǭǡǘǔǈ�{�o�n�i�o�p�{�{�{�{����)�5�`�j�o�p�g�Z�B�)�������������������������¿²§²´¿�����������������˹ܹ���
��������ܹϹù����������Ϲ�Óàåìùý����üùàÓÇ�z�y�zÁÇÏÓ�`�l�u�y�������y�x�l�`�S�R�S�S�\�`�`�`�`��5�A�R�i�o�g�Z�(���ݿȿʿѿݿ�����C�O�\�]�a�\�O�G�C�B�C�C�C�C�C�C�C�C�C�C�������&�+�+�'����������������������²»¿��¿¿²±­®²²²²²²²²²²�/�<�D�H�K�H�<�/�*�/�/�/�/�/�/�/�/�/�/�/�?�3�'�&�!�'�3�=�@�B�?�?�?�?�?�?�?�?�?�?�Z�s���������������������s�Z�8�.�;�A�P�Z�������	�����	���������������������3�@�F�L�N�O�L�E�@�3�+�,�/�2�3�3�3�3�3�3ŠŭŹ����������������ŹŭŤŠŕŔŐŔŠ�M�Y�f�r�s�{�x�r�k�f�Y�M�C�@�>�>�@�H�M�M�������������������������������}�������"�.�<�T�W�V�X�W�T�G�;�"��������"����	�	�	�	����������������������z���������������������z�m�i�b�]�a�m�w�z���������ʽн߽нĽ��������y�o�o�r�z�����h�s�s�j�g�Z�M�7�(����%�+�4�A�M�Z�d�h���������ɺκֺںֺɺ��������������������!�-�3�:�>�=�8�-�&�!����������!��������#�2�A�J�K�A�0�����������������y��������������y�m�`�T�H�?�?�J�T�`�m�y���	���!��	��������������������������������	���"�%�"���	�����������������/�;�H�L�T�Z�X�T�H�<�;�2�/�,�/�/�/�/�/�/�ܻ����������������߻ܻٻػ�DVDbDoD{D~D�D�D{DoDbDVDIDEDFDIDIDVDVDVDV L X 1 V ! V ^ : ' 4 4 0 j C G & L f 9 G / ` V c N * n 4  9 > >  m 6 A J g V "  ? 6 g H # _ d $ 8 , E M Z F   �  �  g  \  �  �      �  �  �  .  �  �  �  1  	  P  (  R  �  z  �  w  �    ~    j  �  I  %  �  ]  /  0  �  �  �  �  0  �    �  �  e    �  2  �  �  <  T  �  N  +  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  GC  �  �  �  �  �  �  z  q  g  ]  Q  B  3  "    �  �  �  �  �  _  U  K  @  6  ,  "        �  �  �  �  �  �  �  �  �  �  ?  a  }  �  �  �  �  �  �  �  �  o  F    �  �  �  z  a  H  �  �  �  �  �  �  �  �  �  �  v  b  M  9  '      �  �  �  �  �    i  �  �  �  �  �  �  �  �  p  A    �  [  �  =  �  �  �  �  �  �  v  i  [  N  ;  (    �  �  �  �  �  u  X  :  \  I  6  !  	  �  �  �  �  �  y  R    �  �  �  u    �  C  
  �  �  �  �  �  �  �  s  Y  ?  %  	  �  �  �  �  p  P  0  r  �  '  f  �  �  �  �  �  �  �  �  }  M  �  m  �  �  7  .            �  �  �  �  �  �  f  H  "  �  �  �  n  !       +  7  B  C  C  B  @  >  <  9  5  1  -  +  (  )  9  I  Y  W     g  �  �  *  P  _  g  Z  7    �  H  �  �  1  �  �  �  �  �  �  �  �  �  �  �  �  }  r  f  W  2    �  �  �  �  j  �  }  w  q  j  c  \  T  M  E  :  ,              *  9  `  [  V  O  G  D  <  0      �  �  �  |  C      �  �  �  �  �  �                          �  �  �  _        �  �  �  �  �  �      �  �  x  #  �  )  R  j  �  �  ^  m  x  �  �  �  �  j  =    �  �  �  �  o  8  �  ~  '   �  7  )      �  �  �  �  �  �  �  �  �  y  l  _  R  E  7  *  n  w  u  t  v  v  s  j  ]  O  �  �  �  ]  &  �  �    :  <  H  �  �  �  �  �  {  H  	  �  �  r  Q  >    �  j  �  l  [  �  �  �  �  �  �  �  �  |  l  U  8    �  �  �  �  t  U  6  t  c  Q  @  .    	  �  �  �  �  �  {  b  Q  ?  -    	  �  �  �  �  �  �  �  �  �  �  �  x  X  7    �  �  �  �  b  <  �  �  �  �  �  �  �  �  x  N    �  �  �  F    �  �  G  �  [  �    &  /  /  %    �  �  �  w  +  �  O  �  �  �  �  �  j  a  X  N  D  2         �  �  �  �  �  �  �  �  p  L  )  	�  
c  
�  
�    4  7  "  
�  
�  
i  	�  	{  �  W  �  �  �  �  �          �  �  t  *  �  �  L    �  �  g    �    �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  u  o  j  d  �      O  w  �  k  A  3    �  �  +  �  ?  �  S  �  �  k  r  _  M  :  (      �  �  �  �  �  �  �  �  �  �  �  �  }    �  �  �  R  �  %  U  d  F  �  _  �  �  c  �  �    �  �  I  G  E  C  @  :  5  0  *  $            �  �  �  �  c  �  �  �  �  �  v  c  Q  >  +    �  �  �  �  �  ~  b  G  +  K  Y  h  v  �  �  �  �  �  ^  %  �  �  �  �  �  �  w  f  U  y  �  �  �  v  a  L  7    �  �  y  (  �  }    �  B  �      �  �  �  �  �  �  �  h  G    �  �  k  2    �  �  �  �           �  �  �  �  �  �  �  n  R  6    �  �  �  �    c  p  y  �  �  �  �  �  �  �  �  N  �  �    w  �  V  �    )  i  �  �  �       +  /  +    �  �  K  �  J  �  �  F    	�  	�  	�  	�  	^  	'  �  �  �  �  �  �  J  �  �    o  �  t  Y  7  "    �  �  �  �  �  �  y  b  H  -    �  �  �    x   �  �  �  �  �  �  �  �  �  |  o  e  _  Y  V  X  [  ^  b  g  k  7  W  v  �  �  �  �  �  �  ^  0    �  �  �  Z  �  |  �    �  �  �  �  �  �  �  �  �  e  F     �  �  s  "  �  C  �  I  �  �  �  �  �  �  �  }  i  \  S  L  @  .       �  �  �  j  !      	    �  �  �  �  �  �  �  }  ^  ;    �  �  �  j  �  e  N  :  $    �  �  �  �  �  \  8    �  �  �  �  �    	�  
  
�  
�    &  0     
�  
�  
|  

  	x  �  8    �  �  �  T  	Z  	�  	�  	�  	�  	�  	�  	�  	�  	U  �  �  6  �  (  �  �  1  �  )  n  i  d  `  \  [  Z  Y  Y  Z  Z  [  `  i  q  z  �  �  �  �  +    �  �  �  �  y  W  8    �  �  �  o  9  �  �    �  :  f  T  A  .      �  �  �  �  �  �  �  p  V  <  #      �  �  �  �  �  �  �  �  ^  1    �  �  Z  �  u  �  ;  �  �    �  A  �  ?  �  `  
�  
s  
  	�  	6  �  W  �  5  Q  1    �  �