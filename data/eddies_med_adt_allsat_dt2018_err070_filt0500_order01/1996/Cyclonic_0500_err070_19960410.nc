CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�������       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�k�   max       P���       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��j   max       <�t�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @F��Q�     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @v{33334     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P@           �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�p        max       @��           7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��F   max       <t�       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B1zr       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B0�|       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >.�   max       C���       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >1��   max       C���       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          e       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          C       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          ?       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�k�   max       P�A�       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�E����   max       ?��x���       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       <�o       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @F�          
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v{33334     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @P@           �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�p        max       @�$@           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B   max         B       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�6��C-   max       ?���f�A�     �  \8   	                     d                           G               !         '   %               *   $      >      ]                        
      	                                 !                      :               NَOR�M�k�N�!4N���N��P�;P���O,~O4�O��{OQZ�P�A�N�΁P KO�i�PX]JOq�N�O���N|�VO��LO��NO�$�O�˹P�XOB�SNR��N.^�O`I(O�8�O���N���OԸ�N]/�P#��N} O���N`g�N(A,O���Ob-�N�pO�NP w?N��N���NTf�PB�PPf�O�"N<B�N�X�N���O�}�OD��O O7�LO��1Np�N{:sN� tN��Of�NN���N���O@˰O;��N�k<�t�<u<D��<o;��
;�o�D����`B��`B�49X�D���D���D���e`B�e`B�u�u��o��C���C���C����㼣�
��1��9X��9X��j��j��j�ě��ě����ͼ��ͼ�/��/��/��h�o�o�o�C��C��C��C��\)�\)�t��t���P��P��P��P�49X�<j�<j�<j�<j�D���D���H�9�L�ͽixսu�}󶽏\)��t���t����-��jggtz��������tpkilggg
!#/145//#
� zzz������zzzzzzzzzzzvz���������zupvvvvvv./7<HLU[]UH</-......��������������������,6?BOV[hmnsq[OB0)!$,��)B<Onl[O6����?CLO[]ahmnheSOB=?<>?ABN[ghssnjg[XNJBB=<A%(-/<HOacZUH9/#+/<>Han|��znaYUQHB1+��#8n{��nmt^P<#���otx�����������~wtqoo��������������������NU[g�����������tg[ON���������.97+������
/5BG=:595)O[`hiqtvthg[OKHFIOOONTX[agt������}xtfOMN����������������������������������������
#/<D=4/
�����������������������������)38720�������!-7COhx�����u\6��������������������()5BBHB;5/)%((((((((����������������������������������������0IUbip|}|ubUMG<0���

���������������������������������5BReaSNB5 ����������������������������6O[eorm[B)�������������������� ���������������������������������������������>N[gt�������tgNGB98>st������������vtsrrs����������������������������������������Z^gt�����������trkcZ��������������������������������������������


������������05B[���������vgNB120��������� ����������������/<@6)���ahjty}th\aaaaaaaaaasz��������{zxussssssTTadjlgda`TTRPPPTTTTaaq���������zm_ZVWUa���������������������)5>BGHB;5)�nz��������������|vqn��������������������RU]annvzznaXURRRRRR�������������������������� �

������#/3643//#���
(./*%#
������
#'()*'$##05<BFGE?<0#������������������������������������������������	���	�������������������������������������������������Ҽ���~�����������������������������������������������������������������������ҾA�>�4�1�3�4�9�A�M�X�V�T�P�M�A�A�A�A�A�A��~�s�k�f�d�f�s����������������������;�$����	��!�;�T�m������������m�T�;�H�<�)��û������л���M�j�t�w�s�����f�H�l�_�S�N�O�S�_�o�x�������������������x�l�����������������ĿѿۿԿѿȿĿ����������<�/�#��
�	�	��
��#�<�H�F�U�[�^�U�H�<�����������������������������������������������b�N�5�8�[�u�������������������Ҿ4�-�(�!�#�(�,�4�A�M�Q�U�Z�]�Z�M�H�A�4�4������f�Z�W�^�R�L�R�N�Z�������Ѿ޾׾ʾ����z�p�e�`�h�q�z�������������������������ʼƼ˼���������������!�.�>�@�8�#���ϼ����������������������������������������ѺL�H�L�V�Y�e�p�r�r�~�~�����~�r�e�Y�L�L�L�(��������������(�5�A�G�N�A�5�(�g�]�\�g�t�t�g�g�g�g�g�g�g�g�g�g�ݿ����������Ŀѿ������
�����������������������������������������������˿y�`�O�?�>�E�G�T�k�y�������ĿпϿɿ����y�Y�:�<�A�N�U�Z�g�s�������������������s�Y�"�	��ʾ������������ʾ׾����%�3�.�"�a�W�U�H�<�1�8�H�U�n�zÇÓ×ÔÌÇ�z�n�a�5�3�,�5�5�B�I�N�P�N�K�B�5�5�5�5�5�5�5�5FFFFF$F0F1F2F1F$FFFFFFFFFF�����������������*�3�C�N�I�C�*����������������н�����2�.����ν������������������������н���׽Ľ���������������������������������������������������ùìääì������������������������ùìçàÝÚàìùüþù÷ìììììììì�a�\�T�I�/�"�����"�/�H�T�g�v�����i�a�ܹѹѹܹ�����������ܹܹܹܹܹܹܹܿ�	����־ʾɾ׾��	�"�5�;�D�M�G�;�.������������ ����	������������������������������������������������������	������	��������[�W�O�T�[�h�tāčĚĢĝĘĕčĆā�t�h�[������*�/�.�*����������������������	���"�&�#�"���	���������w�y�t�r¦¿������������²�o�i�c�o�{ǈǔǘǔǔǈ�{�o�o�o�o�o�o�o�o������������������������������������������������������������������������������ŠŔőŏń�{ŁŖŭ�������������������Š������Ĳĳġĳļ�
��#�1�D�U�z�U�0������s�g�N�A�$��5�Z�h�s�������������������s�����������������������������������������������������������ľ��������������������N�N�K�N�Z�g�s�������������s�g�Z�N�N�N�N���������������������
��
���������������h�_�Y�Z�[�g�h�tāčĜĦĪĩĦĚčā�t�h�t�s�m�l�p�wāćčĚģĦĨĠĚčċā�v�t���������������Ϲ۹޹�����ܹϹù����@�3���������@�L�e�r�������������r�@�m�i�a�[�`�a�m�u�z���}��z�v�m�m�m�m�m�m�-�$�!����!�-�:�E�F�S�_�b�_�S�F�:�-�-�M�C�@�4�'�#�'�4�@�H�M�Y�f�j�i�g�f�Y�M�ME�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E*E EEEE*E7ECEPE\EiEnEuEyE|EuEiE\ECE*����������������(�/�.�(���������������������ûлѻٻлû������������-�%�*�-�3�:�S�_�l�x�|�����x�l�_�S�F�:�-�Q�G�:�.�)�.�:�G�S�`�l�y�}���������l�`�Q�������������������Ŀſѿѿѿ̿ǿĿ����� M  Q M G [ I k > 0 H X Z  M J M R * F < + B b * f Y & 0 g p ) 9 a 5 c 5 v ^ k h 7 " 1 9 0 D o - J K = : k J 3 [ B z P � k j @ m s C T F      M   �  �  �    �    �  �  !  �  r  �  t  c  1        �  
  M  0  �  ^  �  Y  G    �  C  �  R  s  �  x  �  �  k  c  �  6     S  �  �  k  >  �  e  G  �    �  �  �  �  Y  �  �  �  �    �  �  �  �  %<t�:�o<o:�o�ě��D������"Ѽ�`B�+�C���/��P��`B�,1�+�� ż�h�'���/�T���t��<j�y�#�q���8Q��/���'�+�y�#��h��E�����F�,1�Y���w�t��P�`�m�h�t��49X��o�49X�@���w��o�����%�'aG��Y���O߽y�#���w������C��Y��aG���%�����F�������
���罾vɽ�
=B
$BpB��B��BABU�B�PBu�B��B��B�~B�B&�B��B!Z�B
1lB-V�B�tB;bB	��B�MB��B��B+�BK�B1zrBc�BCxBջB B&��B"ϡB �B�hB�gBj�B��B�B@CB�B	uFB
�:B��B�B
�5BB�B$�B	wlB}BܩB	AB�oA��A��AB��B��B�5B�BsBy�B#��B�B��B6�B%B%�hB��B�B
 �B��B��B�VB>BCBA�B�yB��B�MB��B�=B&>�B�fB!�B
{�B-�2B��B?�B	:@B�B.�B�jB*�nB@EB0�|B@�BYBÇB�2B'>�B"��B>TB DB��BG2B��B�3BD>BK�B	��B
��B�eB>�B
;tB9_BC�B29B
7RBu�B<B0aB��A��B 7�B��B�-B1{B�]BG[B�VB#��B �B= B@B$��B%�`B?VB�A���A�ئ@鷾A�c�A<,AF��AiMk@��@���AuaA�A���A��|A:L�AIA���A&YA���?��A�^TA�:�A~�GA�6�An�ZA�r�AX3BA�>MA�=�C���A���A*�9A$�+B9�A��A�'A��=?lgA[�9A�
B1AYr,A��SA��WA[/WA�q�Bi<A�SA�Z�A�ơA�UA���A!��AK/A��#A��A�{�A�9�>.�?�Z�A��@|j@�V�C�6FC���A3F�@�g�@�
�AOAv��A�|�A�0�@�K/AЃA; mAF�Aj�3@� H@�ItAt��A�~LA�g�A���A9�9AJ�uA�=A��A�EE?�n�A���A�z5A~�A��YAn�*A���AU�Aǁ�A��C���A���A*�SA% �By*A�zAĂ�A�z�?��A[�wA�zxB
�AZ�A܁A���A[�A���B�sA�|lA�sA��A�y�A��A!�AK yA���A��nA�}�A݋�>1��?�!A���@{|@��C�)[C��*A3�@��@�&vA�Av��   
                     e                           H               "         (   &               +   %      ?      ]                              
                                 !                      :      	                  
            )   C               ?      +      ;               #      %   #   -               +         #      +      #                     '            +   9   )                        '                                       
            !   9               ?      '      1               #      %      )               )                     #                     '            +   1   )                        '                              N��O
�jM�k�N�!4N���N��DO���PR#;N��FOx�O<��OQZ�P�A�N�Z�OʪO�tP/��O$�N�F�O~�N|�VO��LOT�bO�EBO�/�PvRO'�zNR��N.^�OL�/O� KOk��N���O��hN]/�O��7N} O���N`g�N(A,Ov�Ob-�N�pO�NP w?N��N���NTf�PB�PPDO�"N<B�N�X�N���Or�FOD��O_>O#�O��1Np�N{:sN� tNx�RN�'�N���N���O@˰O,�ZN�k  �  5    �  �    �  �  �  �  T  �  �  _  c  �    �  M  �  �  �  C  �  L  �  �  �  �  z  c  �  �  
�  �  
V  6    Q  �  �     �  	  z  �  G  �  ]  �  O  �  �       �  	a    �  �  �  �    
�  �  �    f  �<�o<e`B<D��<o;��
;D���#�
�ě��T���T����o�D���D����C���t���C��ě����㼓t���t���C����㼼j��9X�����ě����ͼ�j��j���ͼ�h�����ͽ�w��/�ixռ�h�o�o�o�#�
�C��C��C��\)�\)�t��t���P�'�P��P�49X�<j�L�ͽ<j�@��Y��D���H�9�L�ͽixս�%���
��\)��t���t����w��jot�������trmkoooooo
#/134/-#
zzz������zzzzzzzzzzzvz���������zupvvvvvv./7<HLU[]UH</-......��������������������)/16BO]fhmlj[OB61-')���)6D[be[O6���ABCIO[\bhhih[ZOLBAAA=BBN[bgqqolg[ZNLCB>=!)-/<HKUYWUH<5/#+/<>Han|��znaYUQHB1+��#8n{��nmt^P<#���st}��������{utssssss��������������������W[agt���������tg[TRW������*02."������ )58@=:650.)JOR[_hhpthf[OLHFJJJJNOU[bgt�������|wtgQN���������������������������������������
/61/'#
������������������������������)44+�������"/9CO\hu�����u\6"��������������������()5BBHB;5/)%((((((((����������������������������������������'0IYbgmxzwpbUPK<0"�������������������������������������)5BQUQNB5������������������������)6OZ__YOB6)��������������������� ���������������������������������������������JN[gt��������tg[SNIJst������������vtsrrs����������������������������������������Z^gt�����������trkcZ��������������������������������������������


������������05B[���������vgNB120��������������������������/<@6)���ahjty}th\aaaaaaaaaasz��������{zxussssssTTadjlgda`TTRPPPTTTT[am���������zsmiba^[��������������������)05>BFGB954)sz�������������~zxts��������������������RU]annvzznaXURRRRRR�������������������������� �

������ #/1532/*#!      
 #''#
 
#'()*'$##05<BFGE?<0#������������������������������������������������	�	��	����������������������������������������� ������������Ҽ���~�����������������������������������������������������������������������ҾA�>�4�1�3�4�9�A�M�X�V�T�P�M�A�A�A�A�A�A������s�m�k�s��������������������������T�G�;�2�$�"�.�;�T�m�y�������������y�m�T�f�M�:�0���ֻȻ׻����@�^�j�r�o����f�x�n�l�_�W�]�_�l�x�x�����������������x�x���������������������ĿѿѿѿƿĿ��������<�/�#������&�/�<�A�A�L�U�X�[�U�H�<�����������������������������������������������b�N�5�8�[�u�������������������Ҿ4�2�(�%�(�(�4�A�M�P�X�M�A�<�4�4�4�4�4�4������s�f�\�W�W�f���������ξѾ׾ھ˾����z�t�m�h�e�c�u�z�����������������������z�ּɼ��������ؼ����!�.�9�:�6�(���������������������������������������������ѺY�M�L�I�L�X�Y�e�r�}�~�����~�r�e�Y�Y�Y�Y�5�(����������������(�5�A�F�L�A�5�g�]�\�g�t�t�g�g�g�g�g�g�g�g�g�g�ݿ����������Ŀѿ������
�����������������������������������������������ɿy�`�P�G�@�@�G�T�h�y�������Ŀ̿̿ƿ����y�g�Z�D�C�I�Q�Z�g�s�������������������s�g��ʾ��������������ʾ׾�����#�+�"�	���a�Z�U�H�<�=�H�U�a�n�zÇÓÓÒÊÇ�z�n�a�5�3�,�5�5�B�I�N�P�N�K�B�5�5�5�5�5�5�5�5FFFFF$F0F1F2F1F$FFFFFFFFFF����������������*�/�6�C�G�D�C�*������������������н�����,�(���ǽ������������������������Ľн۽ݽ�ݽӽĽ�����������������������������������������������ùìêêò���������������������������ìçàÝÚàìùüþù÷ìììììììì�H�;�/�%�#�"�&�/�;�H�T�^�g�m�q�p�i�a�T�H�ܹѹѹܹ�����������ܹܹܹܹܹܹܹܿ�	����־ʾɾ׾��	�"�5�;�D�M�G�;�.������������ ����	����������������������������������������������������� �	�������	������[�W�O�T�[�h�tāčĚĢĝĘĕčĆā�t�h�[������*�/�.�*����������������������	���"�&�#�"���	���������w�y�t�r¦¿������������²�o�i�c�o�{ǈǔǘǔǔǈ�{�o�o�o�o�o�o�o�o������������������������������������������������������������������������������ŠŔőŏń�{ŁŖŭ�������������������Š����ľĳĿ���
��#�-�7�?�U�f�d�I�0�
�����s�g�N�A�$��5�Z�h�s�������������������s�����������������������������������������������������������ľ��������������������N�N�K�N�Z�g�s�������������s�g�Z�N�N�N�N����������������������
����������������h�_�Y�Z�[�g�h�tāčĜĦĪĩĦĚčā�t�h�t�m�l�p�t�xāĈčĚĢĦħĦĠęčĉā�t���������������ùϹԹܹ��޹ܹϹǹù����@�3���������@�L�e�r�������������r�@�m�i�a�[�`�a�m�u�z���}��z�v�m�m�m�m�m�m�-�$�!����!�-�:�E�F�S�_�b�_�S�F�:�-�-�M�C�@�4�'�#�'�4�@�H�M�Y�f�j�i�g�f�Y�M�ME�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E*E"E*E*E7E;ECEPEXE\EeEhE\EPECE7E*E*E*E*����������������(�/�.�(���������������������ûлѻٻлû������������-�%�*�-�3�:�S�_�l�x�|�����x�l�_�S�F�:�-�`�S�G�;�:�/�:�G�U�`�l�y���������y�w�l�`�������������������Ŀſѿѿѿ̿ǿĿ����� .  Q M G Z 3 i ( , K X Z & P L @ O ) D < + 8 ^ & ^ O & 0 d s & 9 c 5 F 5 v ^ k W 7 " 1 9 0 D o - E K = : k ; 3 X 3 z P � k f 9 m s C N F    �  +   �  �  �  �  �    �  U  �  �  r  �  �  A  B  q  �  
  �  
  �  �  �  �  �  Y  G  �  �  �  �  �  s  �  x  �  �  k  t  �  6     S  �  �  k  >  ]  e  G  �    �  �  d  6  Y  �  �  �  �    �  �  �  }  %  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  c  Q  =  )    .  1  )      �  �  �  �  ^  *  �  �  �  P  !  ,  ^  �    �  �  �  �  �  �  �  s  c  O  9  #    �  �  �  �  �  x  �  �  �  �  �  �  z  s  k  X  D  /    �  �  �  �  �  �  m  �  �  �  �  �  �  �  �  �  z  i  Y  I  @  7  .    �  �  �              �  �  �  �  �  �  �  �  z  a  G  ,     �  `  w  �  �  �  �  �  �  u  M    �  �  t  D    �  h  
   �  
�  4  n  �  �  H  
�  
r  
   	�  	>  �    �  *  d  �  �  �    �    u    �  �  �  �  w  ^  B  "     �  �  �  �  �  �  �  �  �  �  �  �  �  s  N  &  �  �  �  U    �  �  6  �  9  �    6  I  S  R  H  0  
  �    U  ~  m  B    �  J  �    /  �  �  �  �  �  �  �  �  �  �  v  b  E    �  �  f    �  f  �  �  �  ^  9    �  �  �  b    �  �  �  {  h  I  0   �   �  M  T  Z  ^  _  ]  W  M  >  .      �  �  �  �  �  �  r  V  "  ,  \  b  ]  J  /    �  �  {  E  $  �  �  �  X  J  6  	  �  �  �  �  �  �  �  �  s  \  F  2  '         �  �  �  �  �  �    �  �  �  �  �  �  w  F    �  ~  "  �  ?  �  �  �  �  �  �  �  �  �  �  �  ~  q  a  N  7    �  �  �  �  �  �  F  K  D  4  )      �  �  �    ?  �  �  P  �  �    �  �  �  �  �  �  �  �  {  \  9    �  �  �  _  '  �  �  S  &  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  l  �  z  s  s  o  ^  J  0    �  �  �  o  #  �  }  "  �  �  *    "  .  ;  ?  4  %    	  �  �  �  �  �  �  e  5  �  �  �  �  �  �  �  u  j  g  h  U  ?  5    �  �  �  |  H  �  U   �  7  J  L  G  ?  3  "  	  �  �  p    �  I  �  i    �    !  �  �  �  �  �  v  ^  ?    �  �  �  �  p  8  �  �  2  |   �  �  �  �  �  �  �  �  d  <    �  �  �  X    �  1  �  i    �  �  �  �  �  �  �  �  u  j  _  V  L  B  8  +        �  �  �  �  �  �  �  �  {  i  R  ;  "    �  �  �  B    �  �  u  z  y  o  _  J  4       �  �  �  U  +    �  �  �  �  �  P  \  b  b  Y  M  ;    �  �  �  �  S    �  n  �  �  �  5  �  �  �  �  �  �  �  |  d  A    �  �  R    �  O  �  �   �  �  r  c  T  E  5  %      �  �  �  �  �  �  c  B  !      �  
  
m  
�  
�  
�  
�  
t  
<  	�  	�  	T  	H  	}  	9  �    C  a  r  U  �  �  �  �  �  �  �  �  t  Q  *    �  �  �  ]  1    �  �  �  	*  	�  	�  
&  
H  
V  
Q  
=  
  	�  	�  	L  �    J  h  3  x  �  6  3  .  (        �  �  �  �  r  .  �  �  8  �      �            �  �  �  �  �  Y  -    �  �  o  -  �  e  �  Q  <  &    �  �  �  �  �  �  �  z  z  {  R  �  �  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  l  c  Y  O  E  �  �  �  �  �  �  �  �  �  �  �  �  V  )  �  �  �  �  {  i           �  �  �  �  �  p  J    �  m    �  V  �  -    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  t  m  f  	  �  �  �  �  �  �  �  �  h  M  2    �  �  �  �  �  �  �  z  q  a  T  H  6  "    �  �  �  �  u  ;  �  �  F      �  �  �  �  x  l  [  J  6       �  �  �  �  �  r  K  $  �  �  G  8  &                  .  X  Y  U  Q  K  D  <  3  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  f  W  I  :  ]  S  D  $  �  �  e    �  �  �  �  {  �  b  *  �  �  �   �  �  �  �  �  �  �  u  Q  (  �  �  �  \    �  [  �  I  -    O  2    �  �  �  n  C      �  �  �  �  U    �  3  �   �  �  �  �  �    w  m  b  W  M  ?  /      �  �  �  S     �  �  �  �  �  v  `  M  ;  7  )  	  �  �  ^    �  ~  -   �   �         	  �  �  �  �  �  �  �  �  l  V  ?  (      0  G  �  �        �  �  �  �  �  ^  3    �  �  A  �  {    8  �  �  �    _  <    �  �  �  \  -  �  �      �  �  �  �  �  	a  	`  	[  	Q  	D  	0  	  �  �  z  2  �  }    �  �  C  -  �  �  �  �  
  	    �  �  �  �  Y  #  �  �  L  �  �  /  �  r  �  �  �  �  \  .  �  �  �  �  �  X      �  �  �  \      �  �  �  �  �  �  �  �  �  �  �  �  }  o  a  T  G  9  ,    �  �  {  j  Z  M  @  3  +  *  *  *  (  $  !       $  (  ,  �  �    |  u  l  d  [  R  I  3    �  �  �  �  ~  a  C  &  �                 �  �  �  �  �  �  �  �  �  �  �  �  
O  
{  
�  
�  
�  
�  
�  
�  
�  
�  
�  
L  	�  	�  	C  �  �  �  �  ,  �  �  �  �  �  �  �  �  �  �  h  M  5    	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _  6  
  �      �  �  �  �  �  �  y  _  F  ,    
    �  �  �  o    Y  d  b  Z  O  @  .    �  �  �  �  w  U  *  �  �  �  �  �  �  �  �  �  a  9    �  �  �  a  2    �  �  V  	  �  Y   �