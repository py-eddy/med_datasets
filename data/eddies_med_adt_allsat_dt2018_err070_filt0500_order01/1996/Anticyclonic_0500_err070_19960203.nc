CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?���n��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�{>      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��/   max       >	7L      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @E�p��
>     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vm\(�     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @P�           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @��@          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �49X   max       >Q�      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�^]   max       B,��      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B,�0      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?i�@   max       C�PR      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Pz�   max       C�Ik      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          ;      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          3      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       PZ��      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�l"h	ԕ   max       ?��c�A \      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       >O�      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E�p��
>     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vm\(�     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @���          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ov_ح�   max       ?��c�A \     �  N�            9   '   �   "         (         `                        +   U   (            
   	   &      /   	            *      +                        	      
   (   R         ,   N��YO��NM�"O�eOw%>P���O�V�O]�GNI�NO�:�Oa7�N�*"P�{>Nv�O,�*Oo�OKbN]�N��Ox�P�lP.�P#�<P
�M��N���N�uO�}P�{Nl�P~��OoٯN��+N��N[AQPb�dO	�P=�OX�N��O��O(�,O&7N�mN�BmNܮ�NC�N��Oq��O��N��N~=�OR'�N���/��C��D���ě���o��o;o;D��;�`B<#�
<49X<D��<e`B<u<���<���<�1<�1<�1<�1<�9X<�j<�j<ě�<���<�`B<�h<�<��<��=o=+=C�=C�=C�=C�=\)=�P=0 �=D��=D��=D��=H�9=L��=L��=P�`=�%=�\)=�t�=��
=��#>�>1'>	7L��������������������`]^`gy�����������tg`�������������������������������������������trmihmt������#5B[�������gTNB:'";HTaqmjdTH;.eht������������tmheZanz}���zneaZZZZZZZZ��������������������0*+6;BO[fikmsh[OB=60�������������������������!5FNRM@5)�����������������������$-,5<BFNOPNLIB5):<A@IMbjnszwupbUQI<:�����"#���) )+6976.))))))))))LOPX[bfhhmtwtroh][OL~yw���������������~~uwvz��������������zu����������������5N[~��gUN5S\gt�����������wlngS��������������������������������������

������������~����������������)BN[gt���g[B5)")6BCDB>6-) �)5BDRbfV5) ��������������������:31;<<HRU`UNH<::::::mcmnz�������znmmmmmm��������������������������274)������c]^effhty�������vthc�����)6BOW[[VOB������ )+,-,)!����������������������YXZ[]\bht|y}�~uth[YY����������������������������������������878:;?FHTZXWUTPKH;88CAHNT\aba^YTHHCCCCCC#07<C<30%#UH??<44/-/:<?HUUUUUU
 #&&#
�������������������

������������������������������������������zvz���������������z�������������������ĺ��������������������������������������������������������������������������������n�{ŇŊňŇ�{�n�h�i�n�n�n�n�n�n�n�n�n�n������M�U�Y�a�^�K�@�'����ܻ׻ڻ���àÓ�z�r�n�l�g�k�n�zÇÓì������ùìàà�[�tĚĮĮĢĔĄ�t�[�6�#��������6�[�;�T�a�k�t�y�w�h�^�T�L�;�/���
�	��"�;���#�-�4�?�D�F�?�4�(�������������<�H�C�=�<�/�/�/�$�*�/�1�<�<�<�<�<�<�<�<�(�Z�g�}�����������s�g�Z�N�2������(�Ŀѿݿ������ݿѿĿ�����������������ÇÓàãëìïìàØÓÊÇ�ÀÅÇÇÇÇ���<�d�tŏŞŕ�{�]�U�I�0�
������������������'�(�(�(���
������������G�T�`�m�y��������y�p�m�`�Y�T�G�?�>�?�G�����������������ʼ������������~��������:�F�S�V�_�l�o�s�l�i�_�S�F�:�-�+�%�-�.�:�l�x���������������x�l�l�l�l�l�l�l�l�l�l�H�N�U�a�g�n�z�{�z�n�e�a�Y�U�H�E�H�I�G�H�/�<�H�N�U�P�H�B�<�0�/�)�#����#�,�/�/Šŭ������������������ŹŠł�|�ńŖŚŠ�zÓù������������������ìÇ�z�d�Y�T�a�z��/�;�H�T�Z�U�T�Q�/���������������������������������f�A�4�,�)�-�A�Z�f�w����������������������������������������Ļ���������������߻߻������������������������������������������������������������������������������|�w���������(�7�B�N�]�[�L�M�5�(���ܿѿԿ߿��ÇÓÖÛÔÓÇÁ�z�p�z�}ÇÇÇÇÇÇÇÇƚƳ����$�0�7�>�a�N�0��ƳƚƁ�c�_�eƁƚ�T�`�n�r�j�d�_�T�G�;�.�"����"�.�G�J�T���������
�������������������������������
�������������������������������������������������������������������������"�;�F�H�>�4�����������������������!�-�:�F�S�_�e�c�_�Y�S�G�F�:�5�-�!���!�������ʾ�������� ���׾������������Y�e�r�~���������������~�r�g�Y�Q�I�A�L�Y�M�W�Z�[�Z�T�M�A�:�4�(��(�4�A�B�M�M�M�M�'�4�@�H�M�Y�f�k�q�f�Y�M�@�4�'�&��"�'�'�������'�,�/�'�����������������������������������������������~�y����
��#�(�0�<�?�G�<�0�#���
�����������b�n�w�{��{�t�n�h�b�U�K�U�U�b�b�b�b�b�b�f�s���������������������y�s�r�f�d�\�f²®¦¦ª²·¿¿¿»²²²²²²ǡǭǸǺǭǪǡǔǌǍǔǞǡǡǡǡǡǡǡǡ������������
����������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������������������������|����������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͼ4�M�R�Y�f�r���r�f�Y�M�4�'���	���'�4E7ECEPESEPEFECE7E4E*E7E7E7E7E7E7E7E7E7E7 - 9 , @ ? 0  p l N . < : 1 8 E < a N ? * 4 < 9 \ V E O ` A d T F Q K / ; E = U N B F r 7 H k S ! G � Q U @  �  s  Z  l    3    %  �  �  �  �  �  ~  w  z  G  Q  �  9  �     �  �  )  �  M  @  �  �  �    �  �  �  �  =  �  �  �  4  �  �  3  �    �  u  �  .  |  �     &�49X�o�ě�=L��=\)>Q�=t�<t�<�t�=L��<�h<�h=�;d<�1<��=o=+<�`B<�/=0 �=��=�"�=}�=8Q�=o=�w=��=��=�O�=,1=��w=,1=H�9=#�
=�P=���=@�=��
=}�=y�#=�7L=}�=���=u=e`B=u=��P=���=�S�>$�/>o>z�>49X>��B"71B
,�B��B"`gB
@CB	�A�^]Ba�B�B�B��B!� B�sBX,B?B'�]B	�Bc�By�B�1B�BӤBh�B_B�B�B҉B
��B"1B�nBlB!r,B'^BtmB@�BA�BQB�BKPB>�B�B!e~B GA��3A�a�B%HB�
BFBȐB�JB,��B��BK4B\�B"?B	�B@LB"?�B
@PB	?WA�~�B��BH+BߠB��B"?�B��BA5B>�B'�B�{B@�BPSB��B��B�B��BC�B�$B�B��B;�B=�B��B4B!��B=�B�\B(mB@�B��B�BD�B@B��B �_B��A���A�i�B%@"B�<BC�B�$B��B,�0BA	B>B@%@H]A�71A�VS@Ê�Aʶ�Aڜ�A�&�A4iOA��A��Az=jA�v�A��RA��VAiU~@�@��C@�j|A�&�A�YA�i#A�t�A���AA�
A��x@�RXA��SA�EA���Aɢ�B�Ac��A��A�nxA�+�A���@~mQAQ7�?��A:ِ@Ӎ?i�@@�p�A��KA��AE�`A���B��A���C��@�9�C�PR@ΚC��b@�A���A��@�C�A���A�~`A��A7ϯAÀ�A�|�Ay,�Aʈ%A�L�A��]Ah�@�!l@��X@��AńxA¬RA�u}A��A��BAB�A��5@���A���A���A���Aɂ�B4�AdϵAҀNA�j`A���A�	,@���AR��?���A<��@Ξ{?Pz�@��A��A�AD��A�~8B�TA��C��3@��eC�Ik@ͳ�C���            9   (   �   #         )         `                        ,   U   (            
   	   '      /   
            *      ,                        
      
   (   R         ,               '      5   #         #         7                        %   )   -   '   
            -      ;               5      %                                                                                       )                        #      #   '   
                  3               3      #                                                NN3�O?��NM�"O��7Oj�O� O�PO]�GNI�NO�oN�.N�*"P$�&Nv�O,�*N���OKbN]�N��N}l�O��)O1�P ~P
�M��N��fN�uO�}O6�Nl�P?�OoٯN��+N�n�N[AQPZ��N�S.O���OX�N��O��O(�,OP�N�mN�BmNܮ�NC�N��OT��O��N��N~=�O)�&N�  �  �  <  A  5  0  s  �  1  �      �  �    �  8  s  .  �  r  	H  \  �  �  '    �  �  �  �  m  3  �  �  �  �  �  C  !  �    �  �  �  r  �  j  "  I  �  :  �  	��j�D���D��<49X<t�=��<49X;D��;�`B<�/<���<D��='�<u<���<�j<�1<�1<�1<�h<�/=y�#<�h<ě�<���<�h<�h<�=8Q�<��=�w=+=C�=\)=C�=\)=��=�w=0 �=D��=D��=D��=L��=L��=L��=P�`=�%=�\)=���=��
=��#>�>O�>	7L��������������������dacgkt����������ytgd����������������������������������������somnrt{����������tssB<<BN[gt}�����tg[NKB"/;HZbb\SH;/"eht������������tmheZanz}���zneaZZZZZZZZ��������������������367BO[[]^[YOBB963333�������������������������)6>CC5)�����������������������$-,5<BFNOPNLIB5)IGFIUbknptqnibUIIIII�����"#���) )+6976.))))))))))LOPX[bfhhmtwtroh][OL���������������������|z{������������������������������������	5N[gvwtgVLB	S\gt�����������wlngS��������������������������������������

������������~����������������)5BBNMJEB=5)"")6BCDB>6-)!)5?EL]`ZP5)��������������������:31;<<HRU`UNH<::::::hnqz�������zwnhhhhhh���������������������������26)������bbhhjt�������|thbbbb����)6BOVZZUOB6������ )+,-,)!����������������������YXZ[]\bht|y}�~uth[YY����������������������������������������878:;?FHTZXWUTPKH;88CAHNT\aba^YTHHCCCCCC#07<C<30%#UH??<44/-/:<?HUUUUUU
 #&&#
�������������������

������������������������������������������yxz��������������|zy�������������������ĺ��������������������������������������������������������������������������������n�{ŇŊňŇ�{�n�h�i�n�n�n�n�n�n�n�n�n�n������1�4�@�B�D�C�@�4�'����������zÇÓàìòùþùìêàÓÇ�z�x�r�x�z�z�O�[�h�t�Ă��z�t�r�h�[�O�B�:�7�9�@�B�O�T�a�l�o�p�g�T�H�;�/�"�����"�/�;�H�T���#�-�4�?�D�F�?�4�(�������������<�H�C�=�<�/�/�/�$�*�/�1�<�<�<�<�<�<�<�<�5�A�N�Z�^�g�k�j�g�]�Z�N�A�A�5�(�"�(�0�5�ѿܿݿ��ݿݿѿĿ����������Ŀʿѿѿѿ�ÇÓàãëìïìàØÓÊÇ�ÀÅÇÇÇÇ�
�#�<�I�U�e�_�V�I�<�0����������������
����'�(�(�(���
������������G�T�`�m�y��������y�p�m�`�Y�T�G�?�>�?�G�����������������������������������������:�F�S�V�_�l�o�s�l�i�_�S�F�:�-�+�%�-�.�:�l�x���������������x�l�l�l�l�l�l�l�l�l�l�H�N�U�a�g�n�z�{�z�n�e�a�Y�U�H�E�H�I�G�H�/�<�H�K�H�H�<�<�/�%�#�"�#�'�/�/�/�/�/�/ŠŭŹ������������������ŹŭŇŀłŇśŠÓàìù����������ùìÓÇ�z�q�k�p�zÇÓ�	��/�H�M�T�P�P�K�;�/�"���	���������	�����������������f�A�4�,�)�-�A�Z�f�w����������������������������������������Ļ���������������߻�������������������������������������������������������������������������������|�w�������������)�+�(�(�������������ÇÓÖÛÔÓÇÁ�z�p�z�}ÇÇÇÇÇÇÇÇƳ������$�+�/�$����ƳƚƎƁ�j�e�oƁƞƳ�T�`�n�r�j�d�_�T�G�;�.�"����"�.�G�J�T���������
����������������������������������������������������������������������������������������������������������"�;�E�G�>�3���������������������-�:�F�F�S�_�U�S�F�>�:�8�1�-�&�%�-�-�-�-�����׾�����������׾��������������Y�e�r�~���������������~�r�g�Y�Q�I�A�L�Y�M�W�Z�[�Z�T�M�A�:�4�(��(�4�A�B�M�M�M�M�'�4�@�H�M�Y�f�k�q�f�Y�M�@�4�'�&��"�'�'�������'�,�/�'�������������������������������������������������{����
��#�(�0�<�?�G�<�0�#���
�����������b�n�w�{��{�t�n�h�b�U�K�U�U�b�b�b�b�b�b�f�s���������������������y�s�r�f�d�\�f²®¦¦ª²·¿¿¿»²²²²²²ǡǭǸǺǭǪǡǔǌǍǔǞǡǡǡǡǡǡǡǡ�����
������
����������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������������������������|����������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͼ@�M�O�T�Y�a�Y�M�@�4�'������'�4�6�@E7ECEPESEPEFECE7E4E*E7E7E7E7E7E7E7E7E7E7 - + , 4 > )  p l / 3 < ! 1 8 > < a N ; & E 8 9 \ P E O ! A V T F V K / = A = U N B B r 7 H k S  G � Q @ @  [  �  Z    2    Z  %  �  *  �  �  �  ~  w  �  G  Q  �  ~  >    _  �  )  �  M  @  {  �  �    �  �  �  �  �  R  �  �  4  �  k  3  �    �  u  �  .  |  �    &  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  �  �  �  �  �  �  �  �  �  z  V  &  �  �  R  �  �  @  �  ~  =  V  r  �  �  �  �  �  �  �  |  h  P  4    �  �  �  y  5  <  7  2  -  $        �  �  �  �  �  ~  c  H  ,     �   �  3  �  �    .  <  ?  '    �  �  �  �  M    �  �  �  S  �  �  �  �    %  1  5  4  ,      �  �  �    {  �  &  j  �  	k  r  �  �  �  �  Z    �    0    �  J  �  n  "  Y  �  �  �    ;  [  m  s  h  H    �  �  c    �  3    �  �  ?  '  �  �  �  �  �  �  �  �  |  q  h  c  ]  V  L  B  7  +      1  7  <  >  @  C  F  K  S  d  �  �  }  y  v  r  n  i  c  ^  d  �     #  G  n  �  �  �  �  t  ;  �  �  O  �  �  8  �  �  �  �  �  �                  �  �  �  �  r  G    �  �       �  �  �  �  n  X  <      �  �  O    �  o    �    �  1  �  �  �  �  �  �  �  �  f  7    �  �  �  <  z  5  �  �  �  �  �  �  �  }  r  e  Y  J  :  *    	  �  �  �  r  ;           �  �  �  �  �  �  �  �  }  U  ,  �  �  �  R    �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  K    �  �  �  8  0  (             �  �  �  �  �  �  �  u  W  .  �  j  s  l  e  [  N  @  '    �  �  �  �  \  2    �  �  �  T  &  .  .  .  .  '        �  �  �  �  �  �  �  �  �  v  \  A  ;  p  �  �  �  �  �  �  �  �  �  �  �  d  =    �  �  f  �  J  e  r  m  T  -  �  �  b    �  �  �  �  w  ;  �  A  v  k     �  /  �  �  �  	  	.  	D  	?  	  �  �  y  7  �  
  �  l   �    H  W  \  Y  O  ?  *    �  �  �  _  &  �  �  {    |   �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  U  ,  �  �  �  �  �  �  �  �  �  �    )  =  P  d  x  �  �  �  2  �  �  �  �  %  &  !    �  �  �  �  x  Y  :      �  �  �  �  \  �  �          �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  `  �  �  �  �  �  �  �  v  Y  <      �  �  �  �  �  x  Q  )  �  �  �  �  �  w  �  �  �  �    ^  3     �  d    �    |  �  {  m  \  K  :  '    �  �  �  �  �  d  8    �  �  n  8  P  V  f  �  v  ]  >      �  �  ~  @  �  �  _     �  �  T  m  a  U  G  9  +    
  �  �  �  �  �  �  o  X  6     �   �  3  3  -  $      �  �  �  �  r  I    �  �  �  B  �  +   �  �  �  �  �  �  �  �  �  ~  r  b  P  =  2  5  8  9  5  1  ,  �  �  �  �  �  �  �  �  �    r  e  X  K  >  1  #       �  �  �  �  `  :    �  �  �  _  D     �  �  ^    �  %  Y   �  �  �  �  �  �  �  �  �  k  L  6  '    �  �  �  f  ?  �  _  {  �  �  |  n  e  [  I  0    �  �  w  -  �  �    �  �   �  C  .    �  �  �  �  �  n  L  -    �  �  �  <    �  e   �  !         �  �  �  �  t  H    �  �  �  [  /    �  �  �  �  �  �  w  K    �  �  �  w  ?  �  �  k    �  :  �  �  �    �  �  �  v  Y  ;    �  �  �  �  [  J  6    �  �  n  5  �  �  �  �  �  �  �  �  m  R  0    �  �  F  �  `  �  $  �  �  �  �  k  V  >  %    �  �  �  i  =    �  �  �  �  �  o  �    s  g  V  D  3    	  �  �  �  �  �  �  j  R  ;  $    r  p  m  d  [  U  P  K  G  A  7  -  "    �  �  �  T  !   �  �  �  z  F    �  �  �  �  �  �  [    �  f    �  S  �  �  j  S  <       �  �  �  �  �  p  D    �  �  �  U  )  �  �             �  �  �  �  [  %  �  �  G  �  q  �  g  �  B  I  C  .    �  �  |  F      �  �  ^  
�  	�  �  Y  �  �  �  �  }  X  2    �  �  �  �  b  H  7  %    �  �  �  �  �  �  :  �  �  @  �  �  K  �  �  b    �  �  J    �  I  �    L  
y  /  �  �  �  �  �  �  �  ;  
�  
/  	?  O  @  &  c  �  �  �  	  �  �  �  8  �  /  �  .  �  -  �  C  �  K  �  �  �  X  *