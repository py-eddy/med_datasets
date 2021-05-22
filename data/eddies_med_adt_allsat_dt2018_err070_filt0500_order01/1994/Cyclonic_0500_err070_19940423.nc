CDF       
      obs    Q   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��G�z�     D  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�!�   max       P�;�     D  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �Ƨ�   max       <T��     D   4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @F������     �  !x   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @v�p��
>     �  .    effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @R�           �  :�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�}@         D  ;l   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       ;ě�     D  <�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B B�   max       B4�L     D  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4�R     D  ?8   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >3�Q   max       C��X     D  @|   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?OP   max       C��
     D  A�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i     D  C   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          M     D  DH   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          A     D  E�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�!�   max       P���     D  F�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�g��	k�   max       ?�֡a��f     D  H   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �Ƨ�   max       <t�     D  IX   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��G�{   max       @F������     �  J�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @v�p��
>     �  WD   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @R�           �  c�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @��`         D  d�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?7   max         ?7     D  e�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�z�G�{   max       ?���*0U     P  g   1            i                  A   <      
                           
                                                &                  %            
         	      !   
      
   2      	   
   &         
   B                                             OH٠N���N8'�NxP�;�N�g�N#�O�V�O@4NnOLP��BP�O�5^N?�N�9�N���Oc�OL*�OV��N	��N�L�NĹOROO�`sO�+�O&�O��N/�N_�O)�O(k�M�!�ODf�O���OXJSO��Oz��O`��O��Nw\�O<!O&
qN��N���O���O9<�N��xO��9O*v�NҬN��fN��ZN.�5O�B=N��Nv�8N��QO��P�N�Q*N��{PgfO{�O-H�O��O��Nd�.N�Z9O��N�uO�6HOs��Nh��N���NT�O+�TNd�N�Q^N�eN�N NJ�-<T��<49X;ě�;��
:�o%   �o�D���D���D����o��o�ě���`B�o�o�o�t��#�
�e`B�u�u��C���t���t���t����㼛�㼬1��1��1��1��j��j�ě����ͼ��ͼ��ͼ�����������������/��`B��`B��`B��`B��h��h��h��h���C��\)�\)��w��w��w�#�
�'''',1�,1�,1�0 Ž49X�8Q�<j�<j�H�9�T���T���]/��%��C������\�\�Ƨ�y|����������������}y������������������������������������������������������������`rz������������znX`��������������������������������������������
/5RTSH<#
���
#(/,%#
�������������������������#In{�����bI<����x�����������{spx4<Ubns}��{ubUIA:524���������������������$����������05=BN[]^\[NB95000000���

	�������������� ������4>BLN[`eltwvtg[NB644�����������������������&���������������������������>NPW[]_a```[UOB<98<>��)08<?CB@51���pv��������������woopJ[ahs����������th[OJ����������������$/;<?@C</'$$$$$$$$$$�#���������������������� ����������������������FHIUant{��zvnaUMIHF��������������������FKU[anz����~znaUHDDF��������������������;IUbjnstqpbUIECC@<9;!*5COVYYUKC6*'hmz�����������zmgcch�����������������������������������������������������������������������������������������������������
#/771-/-
�������
#%+/5<>6/#
�����������������������%&)BFNX^aa\NB5t~����������������tt����������������������������������������46BHOPSTOB<76-444444�����������������������)53���������#'02<<?@=<0,&# MNUZ[gihggd[NMMMMMMM��������������������(.0/-6B[hz��|th[B1*(���
#<U]]b_gaI#
������������������������������������������������)B[hz��|t[6��	#0<IUZ[YUI<2#
	MP[^gt{�������tg[WNM�������������������������	������������������������������v{�������������{yvv��������������������w���������wwwwwwwwww������������������������������������)6<?<6)KO[hqlhh`[XOECKKKKKK&)**) `git�����������tgf\`acegaUSOMNUaaaaaaaaaZ[]htvyywth_[ZZZZZZZ
#&,*# 
	-/<<HJNHD=<0//---+--�������������������ݻлû����������������ûлܻ�����ܻ��/�,�#��#�*�/�9�<�H�U�^�a�f�a�U�O�H�<�/�����������������������������������������4�)�-�4�A�M�P�M�H�A�4�4�4�4�4�4�4�4�4�4�����g�b�{�l�\�_���ֻ-�S�n�u�m�-��ۺɺ��	��������������	��"�!���	�	�	�	�	�	���	�� ��������	�����������5�4�)������'�)�6�B�O�[�e�S�P�O�8�5ùìëâàÓÏÓàìùý��������������ù����������������������������������������������`�G�5�9�U��������������������������y�`�M�L�T�m�y�����Ŀѿݿ����-���������������������ʼּ����	�����ּʼ�����������
������������ﾌ�������������������������������������������������������������������������������o�g�b�V�L�V�b�o�v�{ǔǡǣǭǭǥǡǔǈ�o����¿²¦²¿����������������ƧƚƘƎƌƇƎƚƧƳ����������������ƳƧŔŉŒŔŠŧŭŰŭŠŔŔŔŔŔŔŔŔŔŔ�N�J�B�;�<�A�B�N�[�\�g�l�r�l�g�[�N�N�N�N�Z�T�M�E�A�@�A�M�N�Z�f�h�f�a�Z�Z�Z�Z�Z�Z�m�`�`�m�r�y�������������������������y�m�N�I�C�C�A�F�N�Z�g�s�������������y�s�g�N�Ŀ��������������ѿ����������ݿĿ;�4�"��	����������"�.�7�A�D�D�?�;���������y�m�o�y���������Ŀǿѿݿؿ������H�@�<�;�<�H�U�\�^�U�H�H�H�H�H�H�H�H�H�H��������������������������������������������ŹŭŠŔŞŠŰŹ������������������˾�׾Ѿ׾����	��"�%�/�0�.�"� ��	����FJFHFDFIFJFVFYFWFVFMFJFJFJFJFJFJFJFJFJFJ�����������������ùϹܹ����Թù������(�"�	����սʽǽнսݽ�����(�9�<�:�(����������������������������������������������������������������������������!����!�%�-�:�S�_�|���x�l�_�S�F�:�-�!��ܾ׾оʾľȾ׾����	������	�����������������(�5�E�O�Q�J�A�5�(��#�����#�0�<�I�>�<�0�#�#�#�#�#�#�#�#�����
����*�6�C�H�P�O�L�C�6�*�������x�l�e�S�O�M�S�_�x�~����������������������������!�-�:�?�:�3�-�$�!������	�������	��!�"�/�;�?�A�;�/�-�"���1������)�B�[�r¤ �t�g�N�1²®©¦¥²¿����������������������¿²�����������������ùùϹعܹ�ݹܹϹù�������������������*�6�C�L�M�K�C�6�*�ŒŇ�{�t�n�b�Z�b�n�{ŇŔŠŬŪŨŠŝŖŒ�T�N�M�T�`�g�m�o�m�`�T�T�T�T�T�T�T�T�T�T�Z�Q�N�M�N�Z�Z�g�s�w�|�t�s�g�Z�Z�Z�Z�Z�Z�_�]�Y�_�j�l�x���������x�u�l�_�_�_�_�_�_�ѿʿͿѿֿݿ����ݿݿѿѿѿѿѿѿѿ������������������)�B�O�W�[�X�O�K�B���������(�)�4�A�M�N�V�M�A�=�4�(����)�&�������)�*�5�?�:�5�)�)�)�)�)�)àÕÓÐÍÓàâìïöìàààààààà�Z�M�A�(������(�4�A�Z�l������s�f�Z�л������������лܻ���'������ܻ���ƸƳƯƳƺ�����������������������������s�i�g�d�g�m�s�������������������s�s�s�s���������w�s�z�}�z���������������������������������������������������Ľʽнǽ����	���������������	���!�"�����	�����������	��!�"�.�2�.�.�"���	�����M�@�1����'�4�D�Y�r���ʼӼʼ�����f�M�b�]�V�U�V�b�g�o�{ǆǄ�{�o�d�b�b�b�b�b�b�4�,�'�!�&�'�4�7�@�M�Y�^�Y�N�M�E�B�@�4�4��������Ķķľ�����������!�%���
������������������������������}�������˼����������ʼ������C�F�J�V�Y�e�r�~�������������~�r�Y�L�@�C�ܹ۹ع۹ܹ�����������ܹܹܹܹܹܹܹܹù������ùϹܹ߹����ܹϹùùùùùý����������Ľнѽݽ�ݽнȽĽ������������/�&�#�����#�/�<�H�U�V�U�S�L�I�H�<�/�������Ŀ̿ǿĿ��������������������������ּͼʼ¼Ǽʼּ���������ּּּּּ�E*E(E*E,E7E?ECEPE\E_E\E[EQEPECE7E*E*E*E*EiEaEiEiEtEuE�E�E�E�E�E�E�E�E�E�E�EuEiEi�:�9�5�3�:�G�M�S�W�S�Q�G�:�:�:�:�:�:�:�: 6 \ @ ? < = � i l ] @ E  V 2 ; c d 9 Y / ] J J ! � ] 2 T y ^ � @ o 7 5 ? 2 5 V & w N n F < ) , E O / G ] A a f d   C X  l K H ' | X i +  { 3 9 / O ) W 3 R ] 4    �  �  ;  +  z  �  �  �  W  �    �  c  Q  �  �  v    �    �  X  j  I  �  X  ^  S  C  �  �  m  �  �  �  <  �  �  v  �  �  �  �  �  o  �    s  |    �  �  a  V  �  �  �  �  �    �  5    �  #    �  �  �  y  0    �  �  |  o  |  �  �  �  `�t�;ě�;D��;o��񪻣�
���
��`B�49X�ě���O߽�%��󶼋C��u�e`B��󶼴9X���
��o����C���/�\)�,1��h����/������h�\)�����P�`�T���\)���L�ͽH�9��o���,1�'\)�t�����m�h�,1�L�ͽ�w���t���w�'�C��49X�8Q�D����9X��C��L�ͽP�`������+�y�#�P�`��"ѽP�`�H�9��O߽D����\)��+��+��7L�ixս�-�����j���ٽ�����BkBՋB4�LB�B��BZB!dBɴB=�B7�B&w�B*�B'M�B�]B��B\B�bB+AB�#B�	BtB=�B��B.�B�BH(BVBl�B>�B��B5�B	xBa�B!JBwoB'�B'TUB0MB B�B2B�eB!��B"��B5GB�BG�B�5B�OBLB��BJ�B7SB5jB�6B%�yB�B�BAB%wOB ߲BC1B[B%ԷB	�XB�B�VB�}B)�'BH B
��B-�B!�BP�B=B�*B
0�B�B�OB1�B{B�sB@�B��B4�RB��B�OBA)B!�GB��B�~B?�B&�vB*�B'=B�[B?�B1B��B�YB�aB��BE�B�B��BD�B?�B>�B?�BA�B:B�B@�B?�B@�B!��B;�B �{B'@B0F�A���B'B��B"-*B"�GB@DB��B?>B�B��B@UB��BFBE@BB�B>mB%{�B��B��B=�B$?�B �
BC�B6oB&?�B	��B �B��B��B)��B��B�B-?�B!A�BE)BA�B��B
>�B��B�.B=(B��B�:@���A���AKx*A:b)@I�A��RA�1@A��A͞ A� A�.�AxpA �HA�OAH��AI7�B��A�GB?jA�f{A�4A=ԋApxA��4A}�dA_<�As��A���A��>A��AYiIC��X>Nw�A0��A�ñA��@@��AV��A�ΟA��A�V�@��U@e�ZA��xA���A�NK>3�QA��7A�AhʋA�Z�@�5A|�!A�!?A8OA�A�iCA=9�@��[Bz�A��A���A##�A[A\3�@ޝeB�d@�G�A�8�A�;�@�g?�X?�3>���A'B�A��!Av�A+�C���C���APJ@���A��AJ��A:��@O��A�?�A��Aׄ4A���A�[�A�^6Av��@�k�A�YAG$!AH��B�$A���BJ\A�t�A�UGA=��Ap��A�y�A})kA_aAr�A�tkA���A�gAY�C��
>?OPA.�sA��IA�`�@t�AWA�~QA��PA�}@�*@c�A���A��pA��:>wzA�}�A�}wAi�A�9 @��kA|�kA� �A7xA��<A˘sA>	�@��$BBHA�^�A��A!�AZ��A[,@݂�Bï@���A�sA��A��?���?HQ>��A'NA��At��A"�C���C���A7�   2            i            	      B   <                                                	      	                           '            	   	   &                     	      "   
      
   2      
      '         
   B   	                                                         M         !         ?   ?                                                   
               !                                 '                           %            #   /         +            /         #      '                                             A                  =   1                                                   
                                                                                          /         %                     #      '                              O ��NZnBN8'�NxP���NX4-N#�O|z�O@4NnOLP���P1OO?�0N?�N�9�N���N�)FOL*�OV��N	��N��NĹOROO�`sO�KN�>�O��N/�N_�O)�N�`bM�!�O��Ot�<OXJSO��Of��O�mO�>(N4ZVO<!N��NO�/NL7�O�iNNΦ�N��xO��7O*v�NҬN��fN��ZN.�5O��0N��Nv�8N��QO�aP�N�Q*N��{P$LOm O-H�O��N��'Nd�.N�Z9O��N�uO�6HO[�NO��N:f�NT�N���Nd�N�Q^N��YN�N NJ�-  	�  �  �    }  }  e  �  �  �  �  1  �  �  A  �    �  �  �    �  O  v    �  p  �      o  Q  �  �  w    �  �  ;  �  '  .  �  �  u  -  {  �  �  {  q  A  �    �  �  �  �  P    �    {  7  }  
G  #  ,  �    E  �    �  �    �  !  	�  E   :�o<t�;ě�;��
������o�o���
�D���D����`B����e`B��`B�o�t��#�
�t��#�
�e`B��o�u��C���t���j��1���㼛�㼬1��1��j��1��/�����ě����ͼ���������/�����������t��t���`B����h��h��h���C��0 Ž\)��w��w�T���#�
�''0 Ž,1�,1�,1���-�0 Ž49X�8Q�<j�<j�L�ͽY��e`B�]/��C���C������ě��\�Ƨ��������������������������������������������������������������������������������nz����������������kn�����������������������������������������
/2NRPE</#
������
#(/,%#
�������������������������#Ibn{�����nbI<���y|����������������{y<<IUbnotuqkbUIC><;8<���������������������$����������?BN[ZXNB;6??????????���

��������������� ������4>BLN[`eltwvtg[NB644�����������������������"���������������������������>NPW[]_a```[UOB<98<>��)08<?CB@51���su��������������|wsshht}������uth_adhhhh����������������$/;<?@C</'$$$$$$$$$$�#���������������������� 
����������������������LUanqxz|}znnma\UOLL��������������������FKU[anz����~znaUHDDF��������������������<IUbinstpoibUIDCA=:<!*,6COPSSOODC6*jmqz����������zmieej�������������������������������������������������������������������������������������������������������
#*)$)'#
�������	
#(//3/#
	�����������������������%),5BNV[_`[NB5$t~����������������tt����������������������������������������46BHOPSTOB<76-444444�������������������������	�������#'02<<?@=<0,&# MNUZ[gihggd[NMMMMMMM��������������������<BO[hmtxzyvqh[O@978<���
#<U]]b_gaI#
��������������������������������������������)B[hw�zt[O6 ��

#0<DIUYZXUI<0
MP[^gt{�������tg[WNM������������������������������������������������������������v{�������������{yvv��������������������w���������wwwwwwwwww������������������������������������)66>;6)NO[hmhg\[YOHNNNNNNNN&)**) ggmt~���������ttgfggacegaUSOMNUaaaaaaaaaZ[]htvyywth_[ZZZZZZZ
#%,#
	-/<<HJNHD=<0//---+--�������������������ݻû��������������ûлܻ����߻ܻлû��<�9�/�(�/�<�C�H�U�X�U�S�K�H�<�<�<�<�<�<�����������������������������������������4�)�-�4�A�M�P�M�H�A�4�4�4�4�4�4�4�4�4�4����������������������-�F�Z�[�F�-�⺽���	�������� �	������	�	�	�	�	�	�	�	���	�� ��������	���������������(�)�6�B�O�[�c�Y�Q�O�N�B�?�6�)�ùìëâàÓÏÓàìùý��������������ù����������������������������������������������b�J�@�@�E�Y�����������������������������p�h�j�������Ŀؿڿ޿���
���꿸���������������ʼּ����������޼ּʼ�����������
������������ﾌ�������������������������������������������������������������������������������{�q�o�b�V�b�o�y�{ǈǔǡǡǫǣǡǔǈ�{�{����¿²¦²¿����������������ƧƚƘƎƌƇƎƚƧƳ����������������ƳƧŔŉŒŔŠŧŭŰŭŠŔŔŔŔŔŔŔŔŔŔ�N�K�B�<�=�B�E�N�Y�[�g�k�p�i�g�[�N�N�N�N�Z�T�M�E�A�@�A�M�N�Z�f�h�f�a�Z�Z�Z�Z�Z�Z�m�`�`�m�r�y�������������������������y�m�N�I�C�C�A�F�N�Z�g�s�������������y�s�g�N�ݿѿ����������ѿݿ�������������ݿ���	��	��"�.�2�;�;�@�;�.�"�������������y�m�o�y���������Ŀǿѿݿؿ������H�@�<�;�<�H�U�\�^�U�H�H�H�H�H�H�H�H�H�H��������������������������������������������ŹŭŠŔŞŠŰŹ������������������˾��׾Ծ׾������	���"�*�"���	����FJFHFDFIFJFVFYFWFVFMFJFJFJFJFJFJFJFJFJFJ���������������ùϹܹ�����ܹйù���������ؽ̽˽нݽ������(�4�8�5�(�����������������������������������������������������������������������������!����!�&�-�:�S�_�l�y�z�s�_�S�F�:�-�!���پ׾о˾վ׾������	�
���	��������������������(�5�A�K�N�G�A�5�(��#�����#�0�<�B�<�5�0�#�#�#�#�#�#�#�#�����
����*�6�C�H�P�O�L�C�6�*���l�h�_�^�Z�_�l�x�~���������x�l�l�l�l�l�l�������!�&�-�-�-�!������������
��"�/�;�>�;�/�"�"���������B�5�%����)�N�g�t�q�g�[�N�B¿·²²¯²´¿��������������������¿¿�����������������ùùϹعܹ�ݹܹϹù����*��������������*�6�A�J�K�I�C�<�6�*ŒŇ�{�t�n�b�Z�b�n�{ŇŔŠŬŪŨŠŝŖŒ�T�N�M�T�`�g�m�o�m�`�T�T�T�T�T�T�T�T�T�T�Z�Q�N�M�N�Z�Z�g�s�w�|�t�s�g�Z�Z�Z�Z�Z�Z�_�]�Y�_�j�l�x���������x�u�l�_�_�_�_�_�_�ѿʿͿѿֿݿ����ݿݿѿѿѿѿѿѿѿ�����������������)�6�=�B�C�D�?�6�)�������(�)�4�A�M�N�V�M�A�=�4�(����)�&�������)�*�5�?�:�5�)�)�)�)�)�)àÕÓÐÍÓàâìïöìàààààààà�4�+�#�"�%�(�/�4�A�M�Z�f�w���|�s�f�Z�M�4�л������������лܻ���'������ܻ���ƸƳƯƳƺ�����������������������������s�i�g�d�g�m�s�������������������s�s�s�s���y�t�{��|�������������������������������������������������������������ýͽĽ��	���������������	���!�"�����	�����������	��!�"�.�2�.�.�"���	�����Y�Q�M�L�M�M�Y�f�m�r�������r�f�Y�Y�Y�Y�b�]�V�U�V�b�g�o�{ǆǄ�{�o�d�b�b�b�b�b�b�4�,�'�!�&�'�4�7�@�M�Y�^�Y�N�M�E�B�@�4�4��������Ķķľ�����������!�%���
������������������������������}�������˼����������ʼ������L�D�E�G�K�L�W�Y�e�r�~���������~�r�e�Y�L�ܹ۹ٹܹܹ�����������ܹܹܹܹܹܹܹܹù¹����ùϹϹܹܹܹڹϹùùùùùùùý����������Ľнѽݽ�ݽнȽĽ������������/�/�#�����"�#�/�<�H�Q�O�H�H�=�<�/�/�������Ŀ̿ǿĿ��������������������������ּͼʼ¼Ǽʼּ���������ּּּּּ�E*E(E*E-E7E@ECEPE\EZEQEPECE7E*E*E*E*E*E*EiEaEiEiEtEuE�E�E�E�E�E�E�E�E�E�E�EuEiEi�:�9�5�3�:�G�M�S�W�S�Q�G�:�:�:�:�:�:�:�: & N @ ? I 0 � l l ] B >  V 2 , _ d 9 Y * ] J J  O ] 2 T y \ � ? j 7 5 8 0 1 H & S 5 b ; Q ) - E O / G ]  a f d  C X  d K H ' 9 X i +  { # = J O ' W 3 T ] 4      l  ;  +  �  n  �  9  W  �  �    �  Q  �  y  
    �    �  X  j  I    �  ^  S  C  �  B  m  Y  *  �  <  �  K  '  Y  �  �  _  |  �      '  |    �  �  a  #  �  �  �    �    �  �    �  #  �  �  �  �  y  0  �  q  L  |    |  �  �  �  `  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  ?7  �  	*  	W  	y  	�  	�  	�  	j  	H  	  �  �  K  �  q  �  g  �  Q  �  �  �  �  �  �  �  �  �  �  �  �  �  }  b  F  +  	  �  �  �  �  �  �  �  �  �  �  �  �  ~  w  o  g  ]  N  ?  /                        #  &  *  -  0  4  -    �  �  �  �  �  M  S  i  u  y  }  r  U  )  ,    �  �  [    �  >  �  q  3  r  u  x  {  ~    �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  c  a  ^  \  Z  W  O  C  7  ,       �  �  �  a  .   �   �  �  �  �  �  �  p  T  -  �  �  |  =  �  �  �  �  X  �  �  %  �  �  �  �  �  }  o  a  M  8  (      �  �  �  �  �  �  e  �  �  �  �  �  �  �  �  �  z  r  i  a  \  ]  _  `  a  b  c  �  �  �  p  ?    �  i    �  o    �  >  �  �  �  )  �  s  �  �      -  /      �  �  �  ~  S    �  �    �  �   �  .  R  |  �  �  �  �  �  �  �  j  F    �  �  �  <  �  :   e  �  �  �  �  �  �  �  �  p     �  ~  )  �  z    �  c    �  A  8  /  $      �  �  �  �  �  �  �  �  �  �  o  �  �  �  �  �  �  �  �  �  �  �  }  z  u  o  i  g  i  l  l  Y  F  3  �  �  �  �  �  �  �  l  J  &  �  �  �  c    �  l     �    �  �  �  j  M  0    �  �  �  �  w  W  8    �  �  �  z  j  �  �  �  �  �  �  x  l  \  H  6  &    �  �  �  �  w  1   �  �  �  �  �  �  �  �  �  �  �  �  �  �                �        �  �  �  �  �  }  _  D  )    �  �  �  m    q  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  O  H  @  9  2  )        �  �  �  �  �  y  D     �   �   d  v  p  c  R  ?  +      �  �  �  �  �  �  b  7  �  �  x  .  �  �  �      �  �  �  �  �  |  W  .    �  �  n     �   �  �  �  �  �  �  �  �  �  �  �  �  p  X  0    �  �  ?   �   �  p  f  V  =       �  �  �  �  q  Q  +    �  �  @     �   j  �  �  �  �  �  �  �  �  �  �  �  �  |  d  M  *    �  �  x    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  9      p  a  Q  @  .      �  �  �  �  T  "  �  �  �  �  �  �  L  9  O  n  m  f  O  .  	  �  �  �  K    �  �  L  -    �  Q  [  e  o  s  _  J  6  %         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  `  1  �  �  X    �  6  �    �  �  �  �  �  �  �  �  �  s  U  4    �  �  n  #  �  k  ,  (  w  k  _  S  M  J  B  8  .  "      �  �  �  �  �  �  v  C    
      �  �  �  �  �  �  �  �  �  �  �  u  h  b  ]  X  �  �  �  �  �  x  �  �  �  �  ~  U    �  �  P    �  a     �  �  �  �  �  �  �  �  �  �  �  �  k  >    �  �  ,  �   �  �  $  ;  5  *      �  �  �  l  *  �  h  �  {  �  ~  U  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  U  ;       '  "        �  �  �  �  �  o  J  !  �  �  �  C    �  W  �  �         �     -  )      	  �  �  �  e    �  t  #  �  �  �  �  �  �  �  �  �  �  }  c  J  2      �  �  �  �  �  �  �  �  �  �  �  �  v  _  B  !     �  �  �  [  &   �   �  \  W  K  n  u  l  [  ;  �  �  �  �  �  k  C    �    j  L  �  �  �  �    ,  #  �  �  �  O  )    �  �  �  Y    �  ]  {  s  j  _  P  ?  -      �  �  �  �  Y    �  �  f  +  �  �  �  �  �  �  x  a  I  ,      �  �  �  �  j  '  �  �  +  �  n  U  6      %  2  :  ;  6  ,    �  �  �  �  �  n  O  {  z  y  x  w  w  v  u  t  t  r  o  l  i  f  c  a  ^  [  X  q  a  P  >  (    �  �  �  �  �  �  �  x  l  d  [  ^  f  m  A  7  ,         �  �  �  �  �  �  �  �  �  �  �  �  �     �  �  �  �  �  �  �  �  y  ]  ?    �  �  �  u  I    �  �  %  r  �        �  �  �  |  K    �  �  O    �  v  4  �  �  �  �  �  �  �  �  �  �  �  y  Y  7    �  �  �  d  c  b  �  �  �  �  z  p  e  O  4    �  �  �  �  �  l  R  D  6  (  �  �  �  �  �  �  �  �  �  �  m  P  2    �  �  �  �  �  g  	  N    �  �  �  �  �  �  �  [  "  �  �  7  �  w  �  �  '  P  8  B  9    �  �  �  �  �  �  �  �  \  $  �  �  i  �   �        
     �  �  �  �  �  �  �  z  b  I  .  
  �  �  �  �  �  �  �  �  �  n  W  =        �  �  �  a  4  �       �  �    �  �  �  �  �  p  F    �  �  j  $  �  �  8  �  )  �  v  z  v  k  U  5    �  �  �  q  8  �    �  �  '  �  9   �  7  3  -  !      �  �  �  ~  X  -  �  �  R  �  g  �  �    }  n  `  N  ;  )      �  �  �  �  �  �  t  `  D  %  �  �  	�  	�  	�  	u  	.  �  	T  	*  �  	�  
C  
  	�  	F  �  ^  �  �  �  �  #      �  �  �  �  �  u  Z  >      �  �  �  H  �  �  5  ,      �  �  �  �  �  �  �  t  ^  P  F  <  2  $       �  �  �  �  �  �  �  �  �  �  �  z  j  U  4    �  �  C    �    
     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  E  )  �      �  �  �  �  v  L    �  �  P    �  �  7   �  �  �  �  �  �  �  i  R  7    �  �  �  I    �  �  �  �  �            �  �  �  �  i  4  �  �  <  �  �  6  �  �  #    +  J  i  �  �  �  �  �  �  |  p  d  X  K  F  >  4  )    �  �  �  �  �  �  �  �  �  �  �  �  ~  p  `  O  ?  /      �             �  �  �  �  �  p  D  
  �  X  �  \  �  U  �  �  �  �  �  �  z  j  Y  F  3      �  �  �  �  �  �  �  !  �  �  �  �  z  E  	  �  �  h  0  �  �  N  �  �  $  �  ^  	Q  	~  	h  	M  	/  	  �  �  �  h  =    �  �  (  �  ,  �  �  P  E  7  *      �  �  �  �  �  �  �  |  b  ;    �  �  �  �         �  �  �  �  �  �  �  �  �  �  x  n  g  a  \  V  Q