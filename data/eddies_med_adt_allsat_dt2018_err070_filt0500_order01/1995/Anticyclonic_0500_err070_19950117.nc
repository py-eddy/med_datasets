CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�j~��"�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N �   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �aG�   max       =�S�      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?p��
=q   max       @EǮz�H     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @v?�z�H     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @O            l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�           �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��o   max       >s�F      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��l   max       B,��      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B,��      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =D�   max       C��B      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =s�   max       C��      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N �   max       O��%      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�s�g��   max       ?�|����?      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �'�   max       >hs      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?p��
=q   max       @EǮz�H     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����
=p    max       @v>fffff     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @L�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @굀          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @D   max         @D      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*0U2b   max       ?�{���m]     �  M�   '   �         	         9            8                   
         T      *      
            !   $            
      \      
         �                                    	Oh�}P���N���N��N�hO!_Ni�PO�N@�/O~X;O���P��-Oɪ N���O㻼O���O2��N�OX`�O\�;O��OR�hO�+O�pN"NIOm?�N��rO�>Og9{O�nN�d�N�H_N��N��O#�
O��OT��Ns�4OD��OTm�PB�NPsN�|N��Np�N��N���N_N2�N �O->�N�=N"x'�aG��u�u�e`B�e`B�e`B��o��o<49X<D��<T��<e`B<e`B<u<�o<�o<��
<��
<�1<ě�<���<���<���<�/<�/<�/<�/=o=o=o=C�=�P=#�
='�='�=<j=L��=P�`=P�`=ix�=u=�%=�O�=�hs=�hs=��
=��
=�1=��=ě�=��=�;d=�S�SVWTSSV[cgpt{��tg[S!$5Ngt|������shB50!������������������������������������������������������������p{}��������������}{p}�������������}}}}}}����������[ZYZ[hkkmlh[[[[[[[[[�����
#/<C<8/#
�������#0<LOH81#
 ���)5A�����������b[NE?)=99=B<BOY[`bkuvqh[O=�tjot}�����������������
#/KPLH9/#����#%/<EU\acVPH</#�����������������������������������������������������-,/<HUanvvohbaUTH<1-��%/CILLF</#����
 #./274//#
 �������� ""������)57<65)�����#&),5BN[^gd[PN<5)#�����������������������")05:<8)��+.+15>BN[cgjmnkg[N5+hhlvz������������znh���"()-)����������������������	�
##%# 
		���!#){zxwx}������������{{���������������������������������VRV[fhhssih[VVVVVVVVmlt���������������tm��������������������xyz���������������zx�"	��������34367BOR[]][OKB63333%'/6<HKTHG</%%%%%%%%��������������������HHHSTaajmfaTRHHHHHHH��������������������#$/$##�����������������������



	����������RRTamz{����|zmhdb\UR�������������������������������������������������
��#�.�<�G�<�9�#�
��������������)�6�F�N�J�C�6��üùòÓ��zÇÛçù������������������������������������������B�N�P�[�a�g�i�g�\�[�X�N�B�9�5�4�5�:�B�B���������������������������������������������������������}�r�p�q�r����������������������������������������������U�a�zÇÏÔÔÑÇ�z�n�a�U�H�B�8�0�3�=�U�4�A�M�Z�^�f�Z�M�A�4�/�3�4�4�4�4�4�4�4�4������*�5�9�5�0�&�������������������������������������f�M�4�.�5�@�M�f��N�U�s�������������Z�A�5��������(�5�N���������	���	�����ʾ�����o�p�t���������������ùɹ͹͹ù����������������������������߿Ŀ������Ŀ���������������������������������������������Z�f�s�{�����z�s�j�k�Z�M�;�4�4�=�A�M�Zìù��������ùóìåìììììììììì��	���(�1�6�7�(� ����������������������������������������������ʾ׿�.�9�.�"������׾��������������ʿm�y�����������y�v�m�`�T�H�G�=�;�H�T�`�m��"�.�;�G�T�d�g�`�T�G�.�"��	�� � �	��s�w�o�m�t�q�i�f�Z�M�M�J�K�P�Q�Z�^�f�j�s�.�0�;�G�G�G�;�.�"����"�,�.�.�.�.�.�.�G�T�`�y���������������z�m�b�T�R�G�B�F�G�������������ܹϹƹɹϹܹ��������<�I�T�[�U�I�B�#������������������
�#�<�U�b�n�{ŀŇōŐŐŇŁ�{�n�b�U�P�I�I�N�U�(�5�A�N�P�T�W�[�a�f�d�Z�N�5�(���� �(�����ûлһܻݻ�ܻлλû����������������������������s�l�f�\�f�s�u������������!�$�.�8�.�!�������������!�-�:�D�A�:�0�-�!���������������������ĽǽͽŽĽ����������z�y�}����D{D�D�D�D�D�D�D�D�D�D�D�D�D{DoDdDgDiDoD{�������ĽƽĽý���������������|�����������'�1�4�4�4�'���
����������������(�!������������������Ŀѿݿ������������ݿѿοĿ������ÿļM�f�r��������������r�Y�@�4�!�� �'�4�M�����	��	�������������������������������������Ǻɺֺʺɺĺ������������������������������������������������"�.�/�2�/�'�"������������ݿ�����������ݿѿοпѿݿݿݿݿݿݿݾ��'�(�-�-�(����������������������¾��������������������������������m�z�z���}�z�u�m�a�`�a�f�m�m�m�m�m�m�m�mǡǢǭǭǭǭǡǔǔǔǚǟǡǡǡǡǡǡǡǡ�������������������ŹŴűŹ����������²¿����������¿²¦¦¨²²²²�t�g�[�U�[�g�t�| U B * G 0 3 r " c $ 6 C m X P  2 A 5 < @ 7  | o 9 G l  ` : 6 ; 5 ?  , 0 p ?  Y 9 S C  & f 4 n \ 4 e      �  �  �  �  h  �  c  s      �  _  �  <  J  ~  .  �  �  N  �  :  �    �  �  -  �  �    �  �  �  d  �  �  |  �  �  �  S  �  �  �  �  �  6  H  S  �    k��o>S�ϼ#�
�#�
���
;o%   =aG�<T��=�w='�=�\)=8Q�=o=,1=+=8Q�<�='�=P�`=�/=aG�=�+=�w=t�=<j=+=u=�o=�7L=@�=#�
=@�=L��=�%>I�=�O�=u=�\)=�->s�F=�+=��T=�-=��-=�^5=\=�-=�/=���=�F>J=�B	jhB�B�aBݩB�B)�B��B�QB��BDB$�aB
PbB_B��B��B��B�UB�qB!��B,�B��BPB��B.hB��B�5Be�B�BWOB|=BS�BFB$��B,=B:B�?B,��B�CB�"Bs5B[�B5�Bb%B��B�&A��lBv�B%sB ��BO�A�B��B�B	B?B��B�B�B��B)�%BĭB��B� BzB%D�B
��B�B:�B��B��B�gB�QB!�B7�B��B?�BWB�dB�%B��BB1�B��B6ZB7BuGB$��BBA�B��B,��B�nB @�B�kBAhB��B=�B�|B�{A���B��B$�IB ��BQ�A���B�HB@A���A��A�~�A�w�A���@�R@�ۉA��1A<4cA�ue@�eXA���AMK =D�A�W�A�D�A?�*A͕�A2��A�y�AT��Ai�AaX�A?��A`��Aj|�?��A�=kA�*A�_~@�o�ADz�A	�p@p��A!��C��BA!G�@��*@�%>A|IK@��IA� b@(8�A.O7A�}A}��A3��ALW?A���B��A�yQA�{�A�
�A���A��A��A���A�R@��@��AƟzA<S�A�|@ݰ<A���AL��=s�A�DA��7A>�AA�~�A2�}A�~�AT�2Ai>A`��A?kAaAkZ�?oBA���A�YA�n�@�I�AD��A�R@o�A#�C��A!��@ɧ�@�[5A|��@�:A���@+Q�A-�A�~�A}1#A3&�AM�A���B�!A��(A�oKA��   (   �         
         9            9                            T      *      
            !   $            
      \      
         �                           	         
      9                           #   =   '      )                  %                     #                                       '                                                                     )   #                                                #                                                                           N�.�O��N���N��N�hN��'Ni�PO~+�N@�/Od̎Oʚ6O��%O|��N@>�N�,�O��O�?N�OX`�N%�zO��tOG��O[�	N�ĻN"NIOm?�N��rO�?#O�wO��EN�d�N�H_N��N�}UO#�
N�6�OT��Ns�4O&��O9�ObrNPsN�|Nuq$Np�N��N���N_N2�N �O->�N�=N"x'  w     [  a  ,  $  ]  	�  �  E  �  �  �  �  �  %  Z  e  �     
�  �    �  �    �  �  |  e    �  ~  �  
    -  -  �  �  �  �  (  �  �  �  q    �    M  )  m�'�=�9X�u�e`B�e`B�t���o<t�<49X<e`B<u=\)<�1<�1<��<�C�<�9X<��
<�1=#�
=t�<���=C�<�<�/<�/<�/=C�=�w=C�=C�=�P=#�
=,1='�=��-=L��=P�`=T��=u>hs=�%=�O�=�t�=�hs=��
=��
=�1=��=ě�=��=�;d=�S�[YY[^gtuz~|ttgc[[[[[4336=BN[gotwxtg[NB<4��������������������������������������������������������������������������������}�������������}}}}}}�����������[ZYZ[hkkmlh[[[[[[[[[������
#/;96-#
������#0JLF@60#
����MKPXb������������g[M>=?BBGO[^ahlpqoh[OB>vw����������vvvvvvvv
 #(/;<><</.##)/<BHX`aUOH</#�����������������������������������������������������7:<HHUYUSH><77777777� 
#/<FHHFB</#
��
#-/164/-#
����������� )3585,)!    #&),5BN[^gd[PN<5)#�����������������������'.59;6)��259BMN[]fgjkhg[NB=82jimx�������������znj���"()-)����������������������	�
##%# 
		��� ������{zxwx}������������{{������������������������������������VRV[fhhssih[VVVVVVVVr�����������������vr�����������������������������������������"	��������34367BOR[]][OKB63333'(/8<HJRHF</''''''''��������������������HHHSTaajmfaTRHHHHHHH��������������������#$/$##�����������������������



	����������RRTamz{����|zmhdb\UR�������������������������������������������
���#�-�&�#��
��������������������������)�.�3�2�)������������������������������������������������������������B�N�P�[�a�g�i�g�\�[�X�N�B�9�5�4�5�:�B�B���������������������������������������������������������|�x����������������������������������������������������a�zÇÉÐÐÊÇ�z�n�a�U�H�>�<�8�B�H�T�a�4�A�M�Z�^�f�Z�M�A�4�/�3�4�4�4�4�4�4�4�4������&�)�5�7�3�.�$��������������������������������f�Y�M�8�1�7�@�M�Y�f��N�Z�g�s�����������������g�Z�N�:�5�5�;�N�������ʾ׾����׾Ǿ��������~�z�~�����������ùƹŹù������������������������������	�����������޿ݿֿݿ޿�������������������������������������������Z�f�s�z�����y�s�f�Z�M�A�?�6�?�A�M�P�Zìù��������ùóìåìììììììììì��	���(�1�6�7�(� ����������������������������������������������������ʾ׾���������׾ʾ��������������m�y�����������y�s�m�`�T�J�G�?�<�I�T�`�m�"�.�;�G�T�Z�^�V�T�G�;�.�"��
���	��"�Z�b�f�m�p�m�f�`�Z�U�M�M�M�N�S�W�Z�Z�Z�Z�.�0�;�G�G�G�;�.�"����"�,�.�.�.�.�.�.�G�T�`�y���������������z�m�b�T�R�G�B�F�G�������������ܹϹƹɹϹܹ��������<�I�P�Y�U�G�<�#�������������������#�<�n�w�{ŀŇňŊŇ�{�y�n�b�U�U�M�M�S�U�b�n�(�5�A�N�S�V�Z�_�c�`�Z�N�5�(�����"�(�����ûлһܻݻ�ܻлλû����������������������������s�l�f�\�f�s�u������������!�$�.�8�.�!������������!�-�:�C�@�:�.�-�!������!�!�!�!�!�!�����������ĽǽͽŽĽ����������z�y�}����D�D�D�D�D�D�D�D�D�D�D�D{D{DvD{D}D�D�D�D��������ĽƽĽý���������������|�����������'�1�4�4�4�'���
�����������
���'� �������������������ѿݿ߿�����������ݿѿĿÿ������ĿǿѼM�Y�f�r�~������x�r�f�Y�M�B�@�:�6�7�@�M�����	��	�������������������������������������Ǻɺֺʺɺĺ������������������������������������������������"�.�/�2�/�'�"������������ݿ�����������ݿѿοпѿݿݿݿݿݿݿݾ��'�(�-�-�(����������������������¾��������������������������������m�z�z���}�z�u�m�a�`�a�f�m�m�m�m�m�m�m�mǡǢǭǭǭǭǡǔǔǔǚǟǡǡǡǡǡǡǡǡ�������������������ŹŴűŹ����������²¿����������¿²¦¦¨²²²²�t�g�[�U�[�g�t�| 6 + * G 0 , r  c  9 : V Z   9 A 5 D ) 6  Z o 9 G o  _ : 6 ; 1 ? # , 0 e :  Y 9 O C  & f 4 n \ 4 e    �    �  �  �  �  �  �  s  �  �  F    �    "  Q  .  �  K  �  �  �  �    �  �    O  i    �  �  �  d    �  |  �  �  �  S  �  �  �  �  �  6  H  S  �    k  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  @D  M  �  �  .  W  m  v  u  m  [  3  �  �  y  '  �    R  T    u  �  i  �  �  �  P  �         �  �  D  w  }  �  }  	�  �  [  X  U  Q  M  H  C  >  8  2  +  %      	  �  �  �  �  �  a  Y  Q  H  @  6  )      �  �  �  �  �  �  �  �  �  �  �  ,  *  )  !      �  �  �  �  �  �  s  X  >  #  
  �  �  "  �         "  #  #         �  �  �  r  >    �  �  z  2  ]  T  J  A  8  .  !      �  �  �  �  �  �  �  �  u  `  K  	0  	}  	�  	�  	�  	�  	�  	u  	Y  	5  	  �  s  
  v  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  &  @  D  <  /    	  �  �  �  v  K    �  �  �  Z    �  |  s  �  �  {  q  g  c  ]  _  [  L  4     �  �  j  Q  7  &    y  �  �  9  l  {  �  �  �  y  W  /    �  �  7  �  �  �   �  7  I  P  [  �  �  �  �  v  Y  8    �  �  �  ]  (  �  �  �  i  ~  x  u  t  z  �  �  �  �  �  �  r  a  K    �  d  �  �  ^  u  �  �  �  �  �  �  �  �  �  �  �  �  s  <  �  �  .  �  $  %  $                 �  �  �  �  `  +  �  �  w  P    R  X  R  K  A  5  !  	  �  �  �  G  �  �  �  �  S  ,    e  P  :       �  �  �  �  ~  W  +  �  �  �  h  2  �  �  �  �  {  f  J  *    �  �  �  �  �  �  �  �  c    �  L   �   `  _  �  �  �  �  �  �  �  �  �        �  �  &  �    U  �  	�  
�  
�  
�  
�  
�  
�  
r  
F  
  	�  	�  	7  �  "  x  �  �  H  �  �  �  �  �  �  �  �  �  }  _  ?    �  �  [  �  o  �  $  o  �  �  �  
    	  �  �  �  �  �  �  e  /  �  �  B  �  e  |  l  �  �  z  �  �  �  �  �  �  u  d  N  5    �  �  J  �  �  �  �  �  �  r  b  Q  A  0    	  �  �  �  �  �  {  a  F  +    �  �  �  �  �  �  s  S  4    �  �  k    �    2  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  d  K  1    �  �  �  �  �  �  �  �  �  �  o  l  a  =    �  �  _  $    �  �  H  Q  ]  i  r  {  y  l  \  K  5    �  �  t  )  �  v    �  �  _  d  b  [  K  0  
  �  �  x  =  �  �  Y  �  u  �  k  �  �      �  �  �  �  �  �  �  �  b  <    �  �  �    Z  7    �  �  �  �  �  �  y  w  w  v  v  v  u  u  u  u  u  u  u  u  ~  q  e  X  I  ;  *      �  �  �  �  �  w  Y  :        �  �  �  �  �  �  �  �    f  M  4    �  �  �  �  �  �  �  v  
    �  �  �  �  �  �  �  E    �  u  N  8  
  �  �  I  �  �  *  �  �  �          �  �  s    �  U  �  
  9  O  I  -  !    �  �  �  �  �  �  s  V  2    �  �  �  @  �  �  $  -  *  (      
  �  �  �  �  �  �  �  r  \  G  4  .  �  �  �  �  �  �  �  �  �  }  W  1    �  �  �  �  s  O    �  �  �  �  �  �  z  `  <    �  �  ]    �  c  �  |    �    �  �  �  u  1  �  h  �  A  }  �  o    s  �  �  [  �  
�  �  1  �  �  �  �  z  n  b  W  L  A  7  ,  !       �   �   �   �   �  (  !      �  �  �  �  �  ~  [  7    �  �  �  V  
  �  M  �  �  �  �  �  �  �  �  �  �  �  �  n  ~  c  �  A  R  a  o  �  �  �  �  {  n  `  P  ?  -    �  �  �  �  �  }  w  q  k  �  �  u  S  1    �  �  �  ~  Y  3    �  �  �  p  H  �  q  q  R  ?  3  &    �  �  �  �  k  E    �  �  �  U    �  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  �  {  L    �  �  u  9  �  �  �  A    �  t  +  �  �  c  L    �  �  �  �  Z  ,  �  �  �  �  U  )  �  �  �  x  \  \  ]  M  2    �  �  �  �  �  �  q  K  #  �  �  �  j  9  �  F  �  )  '  %      �  �  �  �  v  M  )    �  �  �  \  �  �  �  m  }  �  �  t  b  P  <  '    �  �  �  �  �  �  j  R  :  #