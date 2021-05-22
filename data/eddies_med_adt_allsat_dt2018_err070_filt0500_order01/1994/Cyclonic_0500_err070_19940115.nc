CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�������      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <�j      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�ffffg   max       @F������     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v33334     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @O�           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�`          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��S�   max       ;D��      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B G�   max       B5 #      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       B ;�   max       B4¸      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >c�4   max       C��x      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��Z   max       C���      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          2      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�   max       P���      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�y=�b�   max       ?�g��	k�      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <�1      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��\)   max       @F������     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v33334     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @M�           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�`          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�u%F
�   max       ?ˢ�w�kQ     �  V�                                             /            
                   
   	                  
            	         
      #               %               #                     2               O�E�O�I
OCLM�N �/ON�O�O'3iOP�O��4N1_�O�ɛO��O�w�P���N,/�OC[�P�KOVޅO���N���Nc�1O��N��QN�'�N��N�XN�(iN`�6O�L�O�z�ODa�Oz�O�HP�0O"O.�OɪNԮN�r�P,^�N��)OFw�OKP�4O�ڄO!�)N��N2c�O��O��O/
KO7�HO#�2O��NY�}N�PPE[TNyMN�0�N4@KN��OR &<�j<�j<#�
<t�;�o;o�ě��49X�D���T���T���e`B��t���t���t���1��j�ě����ͼ���������/��`B��`B��`B��`B�������o�o�+�C���P��P��������w��w��w��w�,1�0 Ž<j�D���P�`�P�`�Y��Y��Y��ixսm�h�u�y�#�y�#�y�#�}󶽏\)���������P����������'�����)079-)$+)	 ��������������������mnz����zncmmmmmmmmmm $(
_anz�������znfdnlhY_���������������)5BKNQNLDB5)Y[bgtv���������tphZY��������������������abinr{��{nibaaaaaaaa���	�������yz���������������{y���+:@C>5)���������#U{������{U<#���fgtx|{tggdffffffffff��������������������`lo������������mda]`GHU`n������zncaWUQJG�
#'/<N[W@:/  ����������������������xzz������������zxxxxmz�������������zidbm���������������������������� �������
#%$(#
����������������������������������������������������������������������������������������������������imsz�����������zmhhi6BOV[hvtk[OB6..20,/69HU[hnwz���znaUE9449����������������������������������������������������������
#0?;70$��������������������������LNOS[gormgd[NNLLLLLLYgt���������������tY������ �����������Y[gt����������tg]TRY))45<DFGB5))"BNXt���������tg[NB?B���������
����������������������������������������������B6)("%)6;?BBBBBBBBBBs{��������������|vtsrv����������������vrY[]htz�������tnhc[ZY_gt�����������togb\_jnrz���������znaefjj���������������������������������������������������������������!,7HPKB6)����9<IJKII<:699999999999<HUU[_abcaaUSHE<:99inz{���zxqnliiiiiiii���


�����������)*,*)$�������g�N�B�9�F�N�g¦´±«¦��ɺº��ºɺֺ�����!�-�6�-�!������#� ���
�����
��#�/�<�A�B�?�<�0�/�#�H�H�F�H�H�R�U�Y�[�U�H�H�H�H�H�H�H�H�H�H�@�:�=�@�L�Y�\�Y�Y�L�@�@�@�@�@�@�@�@�@�@���������������������������������àÓÍÇ�|�z�s�n�l�|ÀÇÓàú����ùìà�0�*�&�)�*�0�8�=�@�I�V�X�b�b�X�V�I�=�0�0���������������ѿٿ������� ���ݿ����(�&�-�8�>�N�Z�g�s�����������������N�5�(���������������ʼм̼ʼ��������������������������������ʾ׾��"�/�+�����׾ʾ��6�4�)�(�'�%�$�&�)�6�B�O�Z�o�s�h�[�O�B�6�Z�N�Z�^�]�_�g�s���������������������g�Z���������y�W�D�D�[����������������������ƚƙƔƚƧƳƾƴƳƧƚƚƚƚƚƚƚƚƚƚ�Z�Q�O�O�Z�f�s��������������������s�f�ZĳĦĦĲ���������
�.�6�0�$����������ĳ�������������������	�� �(�"����	����¬¦¥²¿�����������
������������¿¬�H�F�A�C�H�U�V�a�n�o�q�n�k�a�`�U�H�H�H�H������������������������ƠƑƇƁƆƎƧ������������ ��������Ơ���������������$�%�$�������������������ƼƸ�������������������������t�i�g�]�g�t�U�Q�U�\�b�n�n�{ŀŀ�{�t�n�b�U�U�U�U�U�UŔŌŇńŇőŔŠŨŭŭŭŪŠŔŔŔŔŔŔ���������������������¾¾������������������������(�5�A�N�g�v�n�Z�N�A�5�����ѿɿ����������Ŀݿ�����������ݿ��������������������	��"�*�&�,�'�"��	���P�H�O�T�`�k���������������������y�m�`�P�����������������Ŀѿ��������ݿѿĿ������������ɺ����!�-�;�>�3�*�!���ֺ����L�L�P�N�Q�V�e�r�~�����������~�r�e�Y�N�L�4�3�-�3�4�;�@�M�Y�\�f�j�p�o�f�Y�M�@�4�4������ܻջڻ���"�'�4�=�>�H�M�@��FFFF#F$F%F.F%F$FFFFFFFFFFF����� ������'�)�/�*�)����������ŝŇ�n�T�M�U�b�{ťŹ����������������ŭŪţŠŜŝŠŭŹ��������źŹŲŭŭŭŭ����������	�����&�%���	�����[�[�Z�[�h�i�tāčĚĢġĚĚčĂā�t�h�[�����������������������*�1�&�����������������z�������ּ��� �$������㼽��޺ɺ������úɺֺ������ ���������������'�1�4�@�B�@�@�4�'���������������������������������������������¦¦µ¿����������������¿²¦�<�/����������
��#�/�<�H�U�Z�^�T�H�<����������������������������������������ĳĬĩĩİĳĵĿ��������������������Ŀĳ���������������������ʾҾ־;˾ʾ�������ā�z�t�h�d�`�h�tāčĚĦħĲĦĤĚčāā����ĿķĿ�����������������������������̻��~�x�s�x���������������������������������l�J�#�3�G�l�������Ľͽнܽݽ��ҽĽ�����������������������FFFFFF$F1F7F=FJFKFJF?F=F3F1F&F$FF���������ùϹܹ߹ܹϹɹù���������������D�D�D�D�D�D�EEEEEEE D�D�D�D�D�D�D�ÓÃ�}�{ÅÇÓàèùÿ����������ùìàÓ e \  r = F W " ~ n V h * ; ^ ? G L 9 K J w 8 3 E k ] ( ? ` 9 - ; I 7 3 * X K 4 J U ] X R [ g n I . V N ; " Y J j C D q ^ / .      <  �  9  ?  �  o  u  }  /  h  �    �    J  �  �  �  �  �  �  S  �    �  �  x  �  �    �  �  j  �  y  K  �  O  �  F  �  �  F  �  �  �  �  a  �  #  �  �  a  6  r  H  �    �  j  �  �;D����o���
;o�o�t��\)�����ě��t���C��+�D����`B��o���ͽt��49X�\)�ixս��+�ixսo��P��P�t��\)�t���%�e`B�0 Ž49X��7L��C��@��u��%�H�9�L�ͽ��P�<j�y�#�]/��t���1��C��m�h���w��hs��-��\)��\)��hs���w��+�����S�������
������j��^5B�B�+By�B(3B�uB
lBR�B�B	�BS�B(gRB+�BDRB,B&o\B	�BB"�B ��B��B�B!v�B oB ��B��B��B>�B�CB�B5 #B΃B*O+B G�B�pB_B"h!B!T
B ��B$G�B(�B�bB�B�tB	��B��B
SB-7EB3�B)ˆB��B�B
�sB#�B
.�B�`B�DB�B `�Bz�B&}TB}BUBb�B�TB��B��B�/B�BIUBTB�SB�UB
@BENB(��BG�BE�B�B&�nB	��B!��B ��B�'B��B!��B ��B ;�B� B�XBB*B��B��B4¸B�B*8�B ?�B�dB>�B"D�B!�B ��B$?�BǗB��B;B�KB	�;B��B
C�B,�[BE�B)��B��B#B
��B��B	�|B� B�FB��B @4B��B&SJBH�B?%BG�BP�A��@E�A�~A�R�?�T�Aќ�A˂CB
�[A~#�A�ݾ@��AV��Aب"A���A�h�B��AB�A��YA�K�A��A�ʉA�dB%�BߩB�"A�t,A���A��ALyA�=YA},A��Am*�Ay5,@N�c?�\�@֚�@��C��PA���A�H$A���AZ�A�^�A�]A @I@�@�VHAЧkA�Z�A�78A���A�2AM@PA��A���@�0/A.LA2��C��x>c�4C�I�A˛�A�|@E[A���Aŀ3?�BA�|�Aˎ�B7�AA��C@�?�AT�{A؀'A���A�}�B��AB9A�cA�z�A���A�O�A��BI�B	�B|/A� A�|'A��hAL�RA�|�A~
yA��iAm�Az �@[�?���@���@� �C��A��A�egA���AZ��A�ZA���A�@H��@�:�A��A�)�A�}A�}A�AM�A݀�A�c@� kA>�A2��C���>��ZC�H�Aˏ0                                             /                               
   
                                
               $               %               #                     2                     #                        +               ;         +      #         !                     !               )         %         1            /   -                                    3                                                            ;         '                                                   %         %                     /   -                                    )               Om��N�q�O21�M�N �/ON�OVD�O'3iO1�nO��N1_�O)s�O%�9Oq��P���N,/�O�XP�!OVޅO9��N���Nc�1O��N��QN���NS kN�XN�(iN`�6O��O_�uODa�Oz�Oh:�P�O"O.�OɪNԮN��O��N��)N�� NǾ�P�4O�ڄO!�)N��N2c�O��N��	O/
KO7�HO�O��NY�}N�PO�|�NyMN�0�N4@KN�j�OR &  i    �  �  Q  �  �  �  [  P    z  y  I  �      �  4  �  �  �  �    L    u  _  �  L  ~  T  �  j  8  �  �  ]  D  �  \  �  ^    H  �    ]  ]  �  j  �  �  �  y  �  �  "  B  N  �  q  �<�1<D��<t�<t�;�o;o�49X�49X�T����C��T�����
�������
��t���1���ͼ��ͼ��ͽ�P������/�+��`B��h��h�������,1�\)�+�C��49X�#�
��������w�#�
�Y���w�@��8Q�<j�D���P�`�P�`�Y��Y���O߽ixսm�h�y�#�y�#�y�#�y�#�����\)���������������������������$)/1,)"
	��������������������mnz����zncmmmmmmmmmm $(
_anz�������znfdnlhY_�����������������)5BKNQNLDB5)^gt����������xtpida^��������������������abinr{��{nibaaaaaaaa������������������������������$6<@;5.)��������#U{������{U<#���fgtx|{tggdffffffffff��������������������am������������zneb^aGHU`n������zncaWUQJG�
#)/<MNH<0#
����������������������xzz������������zxxxxmz������������zlgeem������������������������������������
 ###"#'#
	����������������������������������������������������������������������������������������������������imsz�����������zmhhi6BOV[hvtk[OB6..20,/6:AHUXenqy~~zwnaUHA=:�������
���������������������������������������������������
#0?;70$��������������������������NNQX[gmqmgc[ONNNNNNNrty~�������������}vr������ �����������W[egtv�������tgb[YWW)15:BBCBA5.)%BNXt���������tg[NB?B���������
����������������������������������������������B6)("%)6;?BBBBBBBBBBs{��������������|vts~�������������~~~~~~Y[]htz�������tnhc[ZY_gt�����������togb\_lntz���������zznlill����������������������������������������������������������������&0>C?6)����9<IJKII<:699999999999<HUU[_abcaaUSHE<:99inz{���zxqnliiiiiiii����

������������)*,*)$�������t�g�[�N�B�=�I�N�g¦±¯©¦�ֺպɺƺɺֺ̺�����������ֺֺֺ����
�� �
�
��#�/�7�<�@�A�>�<�/�#���H�H�F�H�H�R�U�Y�[�U�H�H�H�H�H�H�H�H�H�H�@�:�=�@�L�Y�\�Y�Y�L�@�@�@�@�@�@�@�@�@�@���������������������������������ìàÓÐÊÇÀ�z�{ÈÓàìùý��ûùðì�0�*�&�)�*�0�8�=�@�I�V�X�b�b�X�V�I�=�0�0�������¿Ŀѿۿ������������ݿѿĿ��N�H�A�>�E�N�Z�g�s�������������������Z�N���������������ʼм̼ʼ������������������ʾž����ľʾ׾����	�� �������׾��6�3�/�+�*�.�6�B�O�P�[�[�g�f�[�O�K�B�6�6�g�c�c�g�s���������������������������s�g���������y�W�D�D�[����������������������ƚƙƔƚƧƳƾƴƳƧƚƚƚƚƚƚƚƚƚƚ�f�a�Z�S�R�Q�Z�f�s�}�������������s�f�fĳħĳ�����������
��(�/�#����������ĳ�������������������	�� �(�"����	����¼²²µ¿����������������������������¼�H�F�A�C�H�U�V�a�n�o�q�n�k�a�`�U�H�H�H�H������������������������ƨƚƐƎƞƧ������������������������ƨ���������������$�%�$���������ƾƹ�����������������������������������t�l�g�e�g�o�t�t�U�Q�U�\�b�n�n�{ŀŀ�{�t�n�b�U�U�U�U�U�UŔŌŇńŇőŔŠŨŭŭŭŪŠŔŔŔŔŔŔ���������������������¾¾����������������������(�)�5�=�A�K�I�A�<�5�(����ѿͿ����������Ŀѿݿ����������ݿ��������������������	��"�*�&�,�'�"��	���P�H�O�T�`�k���������������������y�m�`�P�����������������Ŀѿݿ�����ؿѿĿ��������������ɺ����!�-�5�4�&�����ֺ��L�L�P�N�Q�V�e�r�~�����������~�r�e�Y�N�L�4�3�-�3�4�;�@�M�Y�\�f�j�p�o�f�Y�M�@�4�4������ܻջڻ���"�'�4�=�>�H�M�@��FFFF#F$F%F.F%F$FFFFFFFFFFF�����������&�)�.�)���������ŹŭŠŜŉŇ�{ŇŎŭŹ����������������ŭŪţŠŜŝŠŭŹ��������źŹŲŭŭŭŭ��������������	��������	�����h�g�^�h�l�tĀāčĘĚĞĚĖčā�z�t�h�h�����������������������*�1�&�����������������z�������ּ��� �$������㼽��޺ɺ������úɺֺ������ ���������������'�1�4�@�B�@�@�4�'���������������������������������������������¦¦µ¿����������������¿²¦�<�0�/�$�#�"�#�/�<�H�M�Q�H�B�<�<�<�<�<�<����������������������������������������ĳĬĩĩİĳĵĿ��������������������Ŀĳ���������������������ʾϾԾʾʾ���������ā�z�t�h�d�`�h�tāčĚĦħĲĦĤĚčāā����ĿķĿ�����������������������������̻��~�x�s�x���������������������������������l�`�R�?�7�J�l�����������ǽɽ�������������������������������FFFFFF$F1F7F=FJFKFJF?F=F3F1F&F$FF���������ùϹܹ߹ܹϹɹù���������������D�D�D�D�D�D�D�EEEED�D�D�D�D�D�D�D�D�ÓÃ�}�{ÅÇÓàèùÿ����������ùìàÓ g 2 ! r = F U " u J V v 1 * ^ ? 2 J 9 Y J w 1 3 . j ] ( ? ; / - ; H 3 3 * X K 9 . U A O R [ g n I . $ N ; " Y J j ? D q ^ ) .    5  �  p  9  ?  �  �  u  �  �  h  �  j  �    J  8  t  �  �  �  �  �  �  �  �  �  x  �  S  �  �  �  �  g  y  K  �  O  �  >  �  *  �  �  �  �  �  a  �  �  �  �     6  r  H  D    �  j  �  �  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  6  h  h  e  [  O  >  &  	  �  �  �  x  +  �  �  G  #  W  N  �  !  X  �  �              �  �  �  q  0  �  T  �    �  �  �  �  �  �  y  \  :    �  �  �  Y  &  �  �  �  t  {  �  �  �  �  �  �  �  �  �  �  �  s  !  �  �  j  D    �  �  Q  <  &         �  �  �  �  �  �        	  
        �  �  �  �  �  �  u  k  i  e  ^  U  K  ;  +    �  �  d   �  y  �  �  �  �  �  �  �  �  l  :  �  �  v  $  �  F  �  *  �  �  �  �  �  �  �  �  �  �    `  =    �  �  �  ]  %  �  �  K  U  U  E  7  ,  '  !    
  �  �  �  �  a  9    �  Z   �  3  ?  2  B  N  H  ;  +      �  �  �  �  _  1     �  b               �  �  �  �  �  �  �        %  5  F  W  h  �  �    /  Q  l  z  u  d  L  0    �  �  �  J  �  ~     n  `  k  s  w  y  y  u  s  p  g  Y  :    �  ]  �  �    �  ^  3  2  3  ?  G  A  7  &    �  �  �  �  �  �  s  \  E  %    �  �  �  f  6    �  �  V    �  y  '  �  h    �  �  �   �    
    �  �  �  �  �  �  �  �  q  Z  C  ,     �   �   �   �  �  �  �    �  �  �  �  �  }  V  )  �  �  �  �  �  �  �  �  �  �  t  `  G  *    �  �  �  w  N  )    �  �  �  n  Q  :  4  ,  %          �  �  �  �  �  �  z  a  L  <  /     �  M  U  _  n  |  �  �  �  r  T  >  5  1  '    �  �  <     �  �  �  �  �  �  o  T  6    �  �  �  \  )  �  �    j  /  �  �  �  �  �  �  �  �  �  �  ~  {  �  �  �  {  i  X  D  0    �  �  �  �  �  �  �  �  �  �  �  \    �  z    �  `  �                �  �  �  �  �  �  �  �  �  �  �  �  �  �    2  J  E  <  1  %      �  �  �  �  u  E    �  �  z  E  �  �    "  C  D  ;  >  G  <     �  �  �  f  (  �  �  ^    u  q  l  g  a  [  U  K  =  0  #      �  �  �  �  u  a  L  _  P  A  2  #      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  V  >  %     �   �   �   �  �  �  �  �  �  B  L  G  ;  *    �  �  �  7  �  n  �  �  y  h  x  }  ~  x  j  X  C  )    �  �  q  8  �  �  s    �  M  T  9         #  *  &      �  �  �  �  �  |  b  H  X  l  �  �  �  �  �  s  `  K  5      �  �  �  �  `  ;       �  6  B  V  `  d  g  b  Y  J  -     �  �  O    �  e    v  �    2  8  5  +    �  �  �  m  6  �  �  U  �  �  K      v  �  �  �  �  t  c  R  B  2  $        �  �  �  �  z  m  `  �  �  �  w  [  ?  #  	  �  �  �  �  X  (    �  o  	  �  ,  ]  S  K  E  =  7  C  I  I  J  K  H  =  ,    �  �  7  �  I  D  ,    �  �  �  �  �  v  Y  =       �  �  �  �  �  i  L  �  �  �  �  �  �  �  �  ~  k  X  A  *    �  �  �    ,  �  J  C  >  :  7  ?  U  Y  [  S  <    �  �  q  7  �  �  J    �  �  �  �  �  }  p  c  U  E  5  #  	  �  �  �  w  F     �  
  5  G  Z  ]  ^  Z  M  :  !  �  �  �  Y    �  �    �  N  
              �  �  �  �  �  r  T  6    �  �  �  �  H  =  -      �  �  �  �  �  �  �  r  C    �  �  \  .    �  �  �  �  `  2    �  �  �  ]  )  �  �  V    �     �   �         �  �  �  �  �  p  N  (    �  �  �  \  :  ,  .  9  ]  \  Z  S  ?  +    �  �  �  �  �  �  j  P  4     �   �     ]  N  o  h  b  Z  7    �  v  %  �  �  &  �  _  �  �    �  �  �  �  �  �  �  �  [  2    �  �  l  -  �  �  d  2  �  �  �  �  �    �  �  �  �  �  j  ]  ?    �  �  h  �    7  ?  �  �  �  �    h  Q  0    �  �  �  |  Z  <    �      �  �  �  �  �  �  y  h  Q  6    �  �  �  t  ?    �  �  w  6  �  �  �  �  �  �  �  �  �  �  �  v  i  V  ?    �  �  m  '  y  v  q  i  \  H  .    �  �  �  h  3  �  �  ?  �    �   �  �  �  �  �  �  �  �  �  �  �  �  t  `  H  1        �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	  �  �        	  �  �  �  �    Y  *  �  �    �  c  �  B  9  1  )  !        �  �  �  �  �  �  �  �  r  ]  H  3  N  9  #    �  �  �  �  �  �  i  P  <  )    	  �  �  �  �  �  �  o  S  5    �  �  �  �  �  _  6    �  �  �  b  8    `  p  \  E  -    �  �  �  �  �  e  ;    �  �  P    �  k  �  s  a  N  ;  (      �  �  �  �  �  j  Y  C  %  �  �  -