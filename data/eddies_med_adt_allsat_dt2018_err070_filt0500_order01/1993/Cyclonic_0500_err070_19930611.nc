CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�I�^5@     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M͠   max       P�z     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <�o     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>k��Q�   max       @E������       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v�fffff       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q            �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �(��   max       <t�     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B-ۚ     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�֖   max       B-�     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�W�   max       C��     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�*
   max       C��     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M͠   max       P4��     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�҈�p:�   max       ?�\����?     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <u     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>k��Q�   max       @E������       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v�fffff       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q            �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�d�         4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?!-w1��   max       ?�Y��|��       cx         $   
   	      /                              	      1      
         (   #   	         #             [      
                     �               	                           	               '   
   	         
      $            $               Nb��N)��Osw~O!}N�g8O���PN�~N�՝N˂�N�ȣOEl�Oh�O�x�Ni.O�a�O�MN��NaU$P��O�t$N�zN��OM�Oݺ�O[ {N�C}O+N+O-��P"ȡO�'NIPO�D4P�zO^�&O��OW<�O�CNK{N��eO&�O�,P{��Nh��N3�)O���OT��Nh|O5�jOf�N�i~O��7O�6N=�-N11O	��NԳqN���O2M�N
�5OrqOf{�N)o�N�[-O��LOU��N��&OH O��MN�GN�O�W4OS:�NvcnOG��M͠N��N2R<�o<49X��o���
��`B�o�o�t��#�
�49X�D���T���e`B�e`B�e`B�u�u��o��o��C���C���t���t����㼣�
���
���
���
��9X��9X��9X��9X��j�ě��ě��ě��ě��ě����ͼ��ͼ��ͼ��ͼ�����h���o�o�o�+�+��P��P��P����w��w�#�
�,1�0 Ž<j�<j�D���H�9�L�ͽT���q���u�y�#�}�}󶽁%��o���㽡�����1������������������������?IQU]aZUIB??????????����������������������� ����������������������������������	"'"	�������o{�������������xsmo��������������������,/3<DHSTQH</,),,,,,,�������������������������

���������������������
#<HU[\aUE</#
  #//94/#"BUen��������zjaVQH>B�������).1)�����t{���������}{ttttttt!	��������������������R_s~���������maTMKMR��
#/(# 
��������������������������
#()(%#
 ���������������������������������������%)6@BGLNDB6)JOht{�������th[WOHKJX[chmtz|}ztmkh[WWUXGNa}��������ziaTKIGG-5Ngqtz�zoe[NB50+-+-JNQ[gsog][XNJJJJJJJJ}�����������������t}�����������������}{�STWakmz���{zmaXSRS��������������������5GN[gt���ytpg_NB92/5��������������������^anouz{zznlea^^^^^^^��������������������aaeflnrz�������znida����������������������������������BBDO[db[OB>9BBBBBBBB������������~~������bht�������������tc`b�������������������������������������������� 
#*/0.%#
��������������������������������������������gt�����������|tgc`]g������������������������

����������)*22)!��"!���1<IUbgmmnpnibUIB<911�����������8BJNSgt~�����tg[MB=8#-030##0<=IOPMII<10#!014<IQUZ[[XUPI<90/.0Q[hlkhc[ZPQQQQQQQQQQdn{�������{nlhdddddd�����
#<FNME0������������������������������������uz|����������������u������������������������������������������������������������PU[cgt|��������tg[NP���
"+/;/#
������� ������������!#*--5?FHJJH</# ����������������//7<=HHU`afa`UPH<///���������������������t¦²¶²®¦�ʼ��������ʼּڼݼּʼʼʼʼʼʼʼʼʼ�����������������#�/�:�9�4�7�/�#��
����ƚƐƑƒƚƧƳ������������������ƳƩƧƚ�нǽʽнԽݽ�����������ݽннннн��"��	����������	��"�/�<�H�J�W�T�H�/�"�����m�_�T�Y�`�{�������Ŀؿ�����ѿ����������s�g�c�]�_�g�s�w�������������������/�(�"��"�%�/�;�H�T�T�L�H�;�/�/�/�/�/�/������������������������������������������������������������������������������ìàÓÎÇÄ�z�w�wÃÇÓÝìõ����ÿùì���!�'�&�.�6�B�O�V�h�v�l�X�\�U�O�6�)��<�7�<�<�H�U�W�U�U�H�<�<�<�<�<�<�<�<�<�<��������������������
���������������$��������$�5�=�I�W�T�M�A�?�:�0�$�нƽƽ˽нֽݽ������ݽннннн��O�M�C�L�O�[�h�h�t�h�[�P�O�O�O�O�O�O�O�O���r�Y�M�B�9�@�Y�f������ʼ׼���ʼ�����	�����������	��"�/�;�H�U�Z�T�H�/�"��������������������������������
���
���;�1�0�0�;�H�T�W�[�T�H�F�;�;�;�;�;�;�;�;��ŹųŴż�������������������������ƾ׾پʾ������������ʾ׾�����	�����FcF]FJF1F$FFFF$F1F=FJFVFcFgFlFrFxFoFc���������������������ĿſɿɿƿĿ��������m�h�V�Y�`�e�m�t�y�}�������������������m���׾ɾ������ʾ������"�6�&��	�����5����ҿҿݿ����A�Z�l�������s�g�N�5�������������������*�=�N�U�L�C�*��������	���	���"�#�"������������������������������	����� ���	��~�X�X�]�r�}�����ɺ��4�8�-����˺����~������ƱƧƤƜƧƳ����������������������u�o�h�\�[�U�V�\�d�h�uƀƁƉƎƑƏƎƁ�u������������������%�"����𿫿��������������������Ŀڿ������ݿ��Z�V�X�Z�g�o�s���������s�n�g�Z�Z�Z�Z�Z�Z�t�n�h�[�Y�Z�[�h�tāčĚġĚčċā�t�t�t���������������������������������������׻l�a�_�S�K�I�Q�S�_�k�l�x�����������~�x�l�ܻ����������'�@�r�������������r�Y�����U�T�U�W�\�a�n�t�w�q�n�a�U�U�U�U�U�U�U�U�6�4�6�=�B�O�P�[�[�^�[�V�O�B�6�6�6�6�6�6�/�#�������#�'�/�<�C�H�U�a�]�H�<�/���������������
��#�+�9�<�H�<�0�!��
���{�t�q�{ǈǔǛǝǔǈ�{�{�{�{�{�{�{�{�{�{�=�7�0�&�$����$�.�0�=�I�K�S�Z�X�V�I�=�b�]�Y�n�x�{ŔŭŹ����������ŹŭŠŇ�{�b����������������������������������������īĬįĳ�����������
�� ���������Ŀī�g�f�Z�Z�Z�g�s�����������������������s�g�����������������������������������������a�U�`�a�n�z�~�z�w�n�a�a�a�a�a�a�a�a�a�a������������(�4�E�F�A�1�(������-�+�%�(�+�-�:�F�N�S�T�S�R�R�Q�F�>�:�-�-����׾;Ͼ׾�����	���	����������ؾ��������	�����	������л̻˻лջܻ���ܻлллллллллн������������������ĽǽͽĽý�����������������ݽ�����4�A�F�M�V�\�Z�M�A�(��ù����ùϹ۹ܹ߹ܹϹùùùùùùùùùü'�$��$�'�4�@�M�N�M�J�D�@�4�'�'�'�'�'�'�	����������������������������������	ùíìãäìùÿ����������������������ù�����������������$�%�+�$��������������������Ŀѿݿ�����ݿԿƿĿ��S�:�!�����ں�����!�-�F�f�r�v�t�l�S�H�F�<�/�*�+�/�<�H�U�a�a�i�a�U�K�H�H�H�H�����������ɺֺֺֺɺɺ�����������������ùìàÓÎÈÇÂ�zÇàìñÿ����������ùED�D�D�D�EEE*ECELEPEUEZEVELECE4E*EE������!�*�.�:�E�:�.�.�!������E�E�E�E�E�E�E�E�E�FFF$F1F?FDF0FFE�E�������������������������E�E�E�E�E�E�E�E|E�E�E�E�E�E�E�E�E�E�E�E����������������������������������������� j C > N * U 6 W  P 0 f L 9 @ � I B G M � 2 D + P [ G v h B W ; > R , Q < ] Z   F s N \ W A 2 3 t [ [ c t M p V I c Z 5 Y & L � % < N Z G i C s ? ~ f ] )    �  M  �  o  �  E  �  �  �  �  �  !  �  1  l  #  �  t  �  �  A  �  �  �  �     �  �  �  C  �    P     c  �  �  �  �  O  -  P  �  }  6  �  c  �  H  �  �  �  �  K  c  >  �  �  L  *    B  �  ?  �  �  �  (  �  M    @  �  -      C<t�;�`B��w�e`B�u��t��aG��e`B��9X�e`B��/�t���P��1����h�ě���1�����P��/��j�t��m�h�]/��`B��/��`B�e`B�]/�ě��,1��l��#�
�+���]/��`B�t����,1�(�ý��o�H�9�e`B�'H�9�H�9��w�T���H�9�#�
�'8Q�D���T���aG��@��e`B��{�m�h�m�h���P��C���C��� Žě�������7L������ͽ�1������1�ě���B�TB&�7BvJB��B"`A���B*��B[zB�<B!?�B*B�B&�BB�B�wB)b�BͯB H3A��B�ABD�BݴBw�B�B��BәB��A�DB�B��BkB��A���BDEB�BňB�~B�B8B��B�LB�jB
� BFxB�B�B�B�BBl7B
GB[B4�B�B �B'Bn*B	�B%(�B%��B&��B[�B(��B$64B�BÄB�bB"�B
�B!�B
wB�BCBFLB-ۚBK�Bo�BA�B&�AB�kB�+B"=`A�֖B*;0By�B�B!y�B B�[B�zB�gB@_B��B)�fB��B >�A�_�B�&BAB?�BJ!B �B�YB�	B�`A�^�B�KB��B@/B2�A���BHB��B�B�uB�#B=�B@KB��B�^B
��BʕB:�B
�nB34B��B��B
?�B��BDB|0B<�B'$GB5�B	_IB%<�B%�FB&�tB?�B(��B#�SB1tBʢBĞB"ECB
��B!��B	�VB�B?�B@�B-�B��BA�A���@��?A��'BA+�yA��VAuJ�A���A�r!AJ9�A�]AA�]�A��A�nKA��B
�A*�^A�gC@��hA�%�A�vA�EA�p�ATG'C��Av�'Am	�AW�zA�!3A�D�A�nA��@-�4B��B�UA�^KAx�A��A�hSA�OH@��@י�A��%A��QA�|�A�_B��B
�OA�\�A��4A�W"A��OA�˺A�EJA4��@zjPAW�AXЇ@��A"�FA5��>�W�@�;�A��A��B��Ay,@@wR�A�|�@-�iA�8-C��6A�&C���A�C��A�+�A��a@�+}A��B�^A,ñA�PAt�rA��OA��6AK��A�L�A�~�A؄�A�r\A��B	�mA+�A�A@�8�A�'A�Y>A�v�A��MAT�C��Aw�Al��AW��A��2A�~�A� �A�~k@4oXB�ZBS�A�ZAx��A�b�A�jA��@�@���AƀA�|A�A�},B�oB
�KA�B�A�oA�_�A���A���A�A50�@| �AV}AX��@�H�A"GA7->�*
@��A�qPA�g�B��Ax��@k�A�{�@3��Á	C��A�0C��QAFC�	A�w�         $   
   	      0                              
      2               (   #   	      	   #   !         [      
                     �               
                           
               (      
         
      $             %   	                                 5                  !         !         +   !            '               -   %         7            !               ?         !                                                         +            %                                                -                           !         %   !                           '            -                           %                                                                  +            !                           N=�>N)��O]��O�gN�g8O���PsmN�՝N�<�N�ȣOEl�Oh�O�5UNi.Ox7CO�MN��NaU$O��O�t$N5w�N��OM�O:��O[ {N���O+N+O-��P 
�O�6�NIPN���P4��OF�2O��OW<�O�$�NK{N���O
�#O�,O�#"N-ݸN3�)Oj/�OVMNh|O5�jOp>N�i~O�xO�6N=�-N11N�RHN��KN���O�9N
�5OrqOAi�N)o�N�[-O��LOozN��&OH O���N�GN�O��O7��N@j,OG��M͠N��N2R  �  H  S  �  �  r  �  [  g  -  �  *  �  �  W  n  �  �  R  v  �  �  :    .  �  �  2  �  �  :  �  �  Z  �  E  9  �  �  -  �  
  a  v  E  �  ?  �  �  X  X  �    �  V  �    �  �  E  �  |  o  4  �  b  }  $    
    	Y  +  �  �  �  �<u<49X�ě��ě���`B�o��t��t��D���49X�D���T����o�e`B��o�u�u��o��j��C���1��t���t��\)���
��1���
���
��/��󶼴9X��h�0 ż��ͼě��ě����ě���/�������ͽ�����`B��h�o�t��o�o��P�+��w��P��P���#�
�#�
�#�
�0 Ž0 Ž<j�L�ͽD���H�9�L�ͽe`B�q���u�}�}�}󶽓t���7L���-�������1������������������������?IQU]aZUIB??????????����������������������������������������������������������	"'"	�������w}���������������ttw��������������������*/09<>HOQOH</.******�������������������������

���������������������
#<HUYWNB</(#
#//94/#"KUanz��������znaZTIK�������).1)�����t{���������}{ttttttt!	��������������������R_s~���������maTMKMR��


���������������������������������
#()(%#
 �����������
����������������������������()6BJLBA6)JOht{�������th[WOHKJX[chmtz|}ztmkh[WWUXRai���������waTOLJKR38BN[aeggc_XNB711223JNQ[gsog][XNJJJJJJJJ����������������������������������������STXamz~~zwmaYTSS��������������������5GN[gt���ytpg_NB92/5��������������������^anouz{zznlea^^^^^^^��������������������hnz�������znlfbfhhhh���������������������������	������:BO[\_[OB@::::::::::������������~~������ejt������������tnfce�������������������������������������������� 
#*/0.%#
��������������������������������������������gt����������vqjgea`g������������������������

����������)*22)!�! ��8<IUbekibUID<;888888�����������<BN[gt|����tg[ONBB<<#-030##0<=IOPMII<10#!/137<IKUXZZYVUNI<30/Q[hlkhc[ZPQQQQQQQQQQdn{�������{nlhdddddd�����
#<FNME0����������	���������������������������uz|����������������u������������������������������������������������������������cgmt��������ytmg_`cc�
 #)/#
 ���������������������!#*--5?FHJJH</# ����������������//7<=HHU`afa`UPH<///��������������������¦²´²¬¦�ʼ��������ʼּڼݼּʼʼʼʼʼʼʼʼʼ����������������
��/�1�6�2�5�/�#��
����ƳƭƧƚƔƖƚƧƳ������������������ƳƳ�нǽʽнԽݽ�����������ݽннннн��"��	����������	��"�/�<�H�J�W�T�H�/�"�����y�h�`�_�i���������Ŀܿ�����꿸�����������s�g�c�]�_�g�s�w�������������������;�0�/�"�!�"�(�/�;�H�Q�Q�I�H�;�;�;�;�;�;������������������������������������������������������������������������������ìàÓÎÇÄ�z�w�wÃÇÓÝìõ����ÿùì���#�(�'�0�6�O�[�h�s�i�[�V�Y�R�B�6�)��<�7�<�<�H�U�W�U�U�H�<�<�<�<�<�<�<�<�<�<�����������������������	�
���
���������$��������$�5�=�I�W�T�M�A�?�:�0�$�нƽƽ˽нֽݽ������ݽннннн��O�M�C�L�O�[�h�h�t�h�[�P�O�O�O�O�O�O�O�O�r�W�E�Y�f�r��������ʼּؼۼټ̼������r��	�����������	��"�/�;�H�U�Z�T�H�/�"�����������������
�
��
�����������������;�1�0�0�;�H�T�W�[�T�H�F�;�;�;�;�;�;�;�;��ŹųŴż�������������������������ƾ�ܾ׾ʾ������ƾʾӾ׾������������FcF]FJF1F$FFFF$F1F=FJFVFcFgFlFrFxFoFc�����������������ĿĿȿȿĿ¿������������m�h�V�Y�`�e�m�t�y�}�������������������m���׾ɾ������ʾ������"�6�&��	���������޿�����(�A�Z�g���}�s�g�N�5��������������������*�6�B�M�C�*���������	���	���"�#�"����������������������������	���	�������������~�q�k�k�{�����ɺ�������
��ɺ����~������ƲƦƧƳ��������������� �����������u�o�h�\�[�U�V�\�d�h�uƀƁƉƎƑƏƎƁ�u������������������%�"����𿸿����������������ĿԿݿ߿����ݿѿ��Z�V�X�Z�g�o�s���������s�n�g�Z�Z�Z�Z�Z�Z�h�]�]�h�tāČĈā�t�h�h�h�h�h�h�h�h�h�h�����������������������������������������l�a�_�S�K�I�Q�S�_�k�l�x�����������~�x�l�M�4�'�����5�M�f�r������������r�f�M�a�V�Y�a�a�n�r�u�p�n�a�a�a�a�a�a�a�a�a�a�6�4�6�=�B�O�P�[�[�^�[�V�O�B�6�6�6�6�6�6�/�#��	����� �%�/�<�H�U�_�[�U�H�<�/�����������������
��"�#�0�/�#���
�����{�t�q�{ǈǔǛǝǔǈ�{�{�{�{�{�{�{�{�{�{�=�7�0�&�$����$�.�0�=�I�K�S�Z�X�V�I�=Ňń�{�p�{ŀŔŠŭŹż��������ŹŭŠŔŇ����������������������������������������įĮĴ�����������
����
����������Ŀį�g�f�Z�Z�Z�g�s�����������������������s�g�����������������������������������������a�U�`�a�n�z�~�z�w�n�a�a�a�a�a�a�a�a�a�a�������������(�4�=�A�B�-�(�������-�,�&�)�,�-�:�F�N�P�N�F�;�:�-�-�-�-�-�-����׾;Ͼ׾�����	���	�����������������	�����	��������л̻˻лջܻ���ܻлллллллллн������������������ĽǽͽĽý������������(�����������(�4�A�C�M�S�Y�M�A�(�ù����ùϹ۹ܹ߹ܹϹùùùùùùùùùü'�$��$�'�4�@�M�N�M�J�D�@�4�'�'�'�'�'�'�	����������������������������������	ù÷ìëìíù������������������������ù�����������������$�%�+�$��������������������Ŀѿݿ�����ݿԿƿĿ��S�:�!����������!�-�:�F�S�e�q�u�s�l�S�H�F�<�/�*�+�/�<�H�U�a�a�i�a�U�K�H�H�H�H�����������ɺֺֺֺɺɺ�����������������àÚÓÏÎÎÓàìøù����������ùìààD�D�EEEEE*E3ECEPEWEUEPEJE7E3E*EED�������!�'�.�8�.�)�!��������E�E�E�E�E�E�E�E�E�FFF$F1F?FDF0FFE�E�������������������������E�E�E�E�E�E�E�E|E�E�E�E�E�E�E�E�E�E�E�E����������������������������������������� Z C > N * U / W  P 0 f R 9 7 � I B E M W 2 D # P [ G v g M W + 5 J , Q 7 ] 0  F > U \ ^ / 2 3 ^ [ a c t M r R I e Z 5 U & L � ' < N Y G i 2 v 2 ~ f ] )    s  M  �  *  �  E  �  �  �  �  �  !  U  1  �  #  �  t  �  �  _  �  �  �  �  �  �  �  �  ^  �    $  �  c  �  a  �  �  /  -    p  }  -  A  c  �  i  �  1  �  �  K    �  �  �  L  *  �  B  �  ?  +  �  �  �  �  M  C  �  U  -      C  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  q  f  H  A  :  3  +  "      �  �  �  �  �  �  �  �  o  ]  L  :  G  R  O  A  +  
  �  �  j  C    �    6  �  �  N  �  {  �  w  |  �  �  }  z  v  q  l  c  U  D  /       �  �  �  n  B  �  �  �  �  �  ~  q  c  Q  ?  '  
  �  �  �  y  N  &     �  r  i  a  Y  P  E  ;  2  )        �  �  �  �  �  m  E    �  �  �  �  �  �  �  �  �  �  y  N    �  �  ,  �  O  �   �  [  S  K  C  ;  2  *  !      �  �  �  �  �  �  �  x  b  M  ]  d  g  g  c  ]  T  I  =  ,      �  �  �  V  #  �  �  }  -  *  '  $  !                          "  $  '  �  �  �  �  �  �  �  �  �  �  �    _  <    �  �  �  4  �  *  &      �  �  �  �  |  N    �  �  �  R  �  �  �  G   �  �  �  �  �  �  �  �  �  �  ~  e  @    �  �  8  �  �  r    �  �  �  �  �  �  �  �  |  l  Z  H  -    �  �  �  |  I    G  H  T  W  S  K  >  .    
  �  �  �  �  f  =    �  �    n  S  8        �  �  �  �  ~  _  A    �  �  P  '     �  �  �  �  �  �  �  �  �  �    l  V  =    �  �  �  c  2    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  r  $  A  P  P  <    �  �  U    �  �  �  �  p  %  �  �  �  �  v  t  p  i  _  ^  c  Z  K  ;  -       �  �  �  �  J  �  d  p  \  H  5  4  u  �  �  �  �  �  �  �  }  Z  4    �  ~     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  X  E  :  (    �  �  �  �  �  �  e  C  !     �  �  �  n  !  �  n  (  �  �  �  �  �          �  �  �  �  U    �  :  �  �  .      �    �  �  �  �  Y    �  z    �  %  �    �  �  �  �  �  �  �  �  �  �  �  �  v  W  7    �  �  �  M     �  �  �  �  �  �  �  �  �  �  �  �  o  M  +     �   �   �   o   L  2  /  ,  +  *  '          �  �  �  �  �  �  p  P  -  
  �  �  �  �  �  �  �  �  �  �  �  �  r  9  �  �  V  �  x  �    C  h  |  �  �  �  x  h  R  ,  �  �  i    �  q    �  "  :  6  2  .  +  '  #            
      �  �  �  �  �    �  �  �  �  �  �  �  �    n  \  F  ,    �  �  �    �  G  �  �  �  �  �  �  �  �  u  L    �  �  '  �  	  4  J  �  H  W  T  J  <  -      �  �  �  �  |  _  A  !     �  �  }  �  �  �  �  �  �  z  m  `  P  ?  .      �  �  �  �  �  �  E  =  5  ,         �  �  �  �  �  �  �  v  X  ;  '      �    *  6  8  /    �  �  �  k  A    �  �  G  �  W  �  h  �  �  �  �  �  v  h  [  M  @  )  	  �  �  �  �  t  Z  @  &  �  �  �  �  �  �  �  �  �  v  c  R  C  A  @  <  3  %  �  �  +  -  -  ,  )  %        �  �  �  �  b  8    �  �  �  W  �  �  �  �  �  d  >    �  �  �  V    �  �  �  E  �  �  @  
�  T  �  A  �  �  �    �  �  _  �  �    
Z  	w  k  $  j  �  .  A  R  _  b  c  a  ^  Z  X  U  R  M  G  @  9  .    U  �  v  l  c  Z  Q  H  >  6  .  &          �  �  �  �  �  �  7  ?  E  B  4          �  �  O  �    �  �  �  �  P    V  n  �  �  �  ^  8    �  �  ~  Q  $  �  �  }    �  B  �  ?  :  4  .  (           �  �  �  �  �  �  r  A    �  �  �  �  �  �    k  U  =  $    �  �  �  Z     �  �  >  �  q  �  �  �  �  �  �  �  �  �  �  �  �  �  z  h  Y  O  F  @  A  X  G  5  $          �  �  �  �  �  j  O  4    )  4  ?  F  P  V  W  U  M  A  0      �  �  �  r  F  $       �  �  �  �  �  �  �  �  �  �  �  �    i  F    �  �  `    �  \          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  O  Q  S  V  M  B  6  '      �  �  �  �  �  �  |  x  u  q  �  �  �  �  �  �  �  �  �  �  �  ~  ]  :    �  �  �  1   �           �  �  �  �  �  �  �  �  N    �  �  T    �  �  �  �  �  �  �  �  �  �  �  �  �  ^  4    �  �  �  U  !  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  E  ;  1  #      �  �  �  �  �  n  Q  3    �  �  �  �  �  {  �  �  �  t  e  L  *  �  �  �  P    �    0  �  c  �    |  ~    y  s  l  e  [  P  D  8  ,         �  �  �  �  �  o  j  d  ^  W  N  D  3      �  �  �  �  m  H  !   �   �   �  4    �  �  �  �  a  L  ;  I  #  �  �  �  I    �  a  �  �  �  �  �  �  �  �  �  �  �  o  [  D  +    �  �  �  �    �  b  1  �  �  �  N  �  �  @  �  ~     �  k    �  I  �  �  ?  }  f  @  !      �  �  �  �  Y    �  �  C  �  �  E  �  �  �    �  �  �  m  2  �  �  �  �    X  *  >  '    �  w          �  �  �  �  �  �  �  �  r  \  4  �  �  l    �    
      $  )  !      �  �  �  �  a  H  /    �  �  �  �  �  7  C  K  h    {  o  _  B    �  �  S    �  :  �  D  &  �  	7  	O  	5  	  �  �  r    �  9  �  e  #  .  �    .  D  P    #  )  (  $    
  �  �  �  �  �  �  w  V  &  �  �  |  A  �  u  8  �  �  \  *  �  j  2  �  �  �  L  �  �  �  q  �  �  �  �  {  t  l  d  ]  U  M  F  :  )      �  �  �  �  �  �  �  �  x  i  \  P  B  -      �  �  �  Q    �  ~  0   �   �  �  3  �  �  �  o  ?    �  �  v  D    �  h  �  �    �  #